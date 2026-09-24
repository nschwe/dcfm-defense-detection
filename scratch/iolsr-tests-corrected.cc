#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/iolsr-module.h"
#include "ns3/mobility-module.h"
#include "ns3/wifi-module.h"
#include "ns3/dsdv-module.h"
#include "ns3/output-stream-wrapper.h"
#include "ns3/netanim-module.h"
#include "ns3/udp-client-server-helper.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/energy-module.h"
#include "ns3/ipv4-routing-table-entry.h"
#include <string>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <fcntl.h>
#include <unistd.h>
#include <sys/file.h>
#include <cstring>
#include <cerrno>
#include <limits>
#include <set>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("StableNetworkRouteMod");


std::vector<uint32_t> macTxPerNode;//uint32_t macDataPkts = 0;
std::vector<uint32_t> macDropPerNode; //uint32_t macControlPkts = 0;

uint64_t g_helloCount = 0;
uint64_t g_tcCount = 0;
uint32_t g_midCount = 0;
uint32_t g_hnaCount = 0;
uint64_t g_totalTcRows = 0;
uint64_t g_olsrControlBytes = 0;
uint32_t g_udpReceivedAtWindowStart = 0;
Ptr<UdpServer> g_udpServer = nullptr; // set in main; lets SaveScenarioMetrics read per-window UDP delivery

// Single FlowMonitor approach
Ptr<FlowMonitor> g_flowMonitor = 0;
FlowMonitorHelper g_flowHelper;

// ===========================================================================
// PASSIVE-OBSERVER INSTRUMENTATION  (revision 2026-08)
// ---------------------------------------------------------------------------
// Adds K passive vantage points. Receive-side only: hooks WifiPhy/MonitorSnifferRx
// on a subset of existing nodes. Simulation dynamics are NOT affected, therefore
// every globally-computed metric remains bit-identical to iolsr-tests-mitigation.cc.
// ===========================================================================
const uint32_t NUM_OBSERVERS = 10;
std::vector<uint32_t> g_observerNodes;   // node ids acting as vantage points

// Running mean/std accumulator (Welford). Used by the v4 listener metrics so a
// window's timing statistics need O(1) memory per tracked entity.
struct WelfordAcc {
    uint64_t n; double mean; double m2;
    WelfordAcc() : n(0), mean(0.0), m2(0.0) {}
    void Add(double x) {
        ++n;
        double d = x - mean;
        mean += d / static_cast<double>(n);
        m2 += d * (x - mean);
    }
    double Std() const { return (n > 0) ? std::sqrt(m2 / static_cast<double>(n)) : 0.0; }
};

struct ObserverCounters {
    uint64_t frames;        // every frame successfully sniffed at PHY (incl. ACK/ctrl)
    uint64_t dataFrames;    // subset that carried an IPv4 payload
    uint64_t bytes;         // total bytes sniffed
    uint64_t tcCount;       // TC messages heard
    uint64_t tcRows;        // sum of advertised links over heard TC messages
    uint64_t helloCount;
    uint64_t midCount;
    uint64_t hnaCount;
    // NOTE (2026-08-03): keys below are FULL 32-bit addresses. An earlier version
    // masked them with & 0xFF, which silently aliased every fictitious node
    // (getFakeAddress() = mainAddress + 65536, i.e. 10.0.0.X -> 10.1.0.X, same low
    // octet) onto the real node that injected it. That erased the defence's most
    // conspicuous observable and inverted the reported signal under mobility.
    //
    // originator -> most recently advertised link count, plus the time it was heard.
    // This is the size of the advertised set as it appears on the air; under an
    // active defence it INCLUDES the injected fictitious link, so it is NOT equal
    // to the node's real MPR-selector count and must NOT be presented as a
    // reconstruction of the global AverageMprCount. There is no denominator that
    // repairs that: dividing by |addrsSeen| only imports the fictitious-address
    // artifact into the ratio. Report SumAdvertisedLinks raw, or the per-originator
    // mean, and state the denominator explicitly.
    std::map<uint32_t, uint32_t> advByOriginator;
    std::map<uint32_t, double>   advTimeByOriginator;
    // Every address the observer has seen advertised or originating a TC. Used
    // as the observer's own estimate of network size, i.e. the denominator that
    // turns the reconstructed selector sum into AverageMprCount. Nodes that are
    // nobody's MPR never originate a TC, so counting originators alone
    // under-counts the network.
    std::set<uint32_t> addrsSeen;
    // source node -> bytes heard from that source, for the per-source rate
    // dispersion that mirrors FlowThroughputStd (50 of 51 FlowMonitor flows are
    // per-node OLSR broadcasts, which a in-range observer receives directly).
    std::map<uint32_t, uint64_t> bytesBySource;
    std::map<uint32_t, uint64_t> framesBySource;

    // --- v4 listener extensions (receive-side only, no dynamics impact) ---
    uint64_t dataBytes;      // bytes of the frames that carried an IPv4 payload
    uint64_t olsrFrames;     // subset of data frames that carried OLSR (UDP/698)
    uint64_t olsrBytes;      // full sniffed bytes of those frames
    uint64_t tcHopSum;       // sum of MessageHeader hop counts over TC heard
    // (originator<<16)|msgSeq of every TC heard: de-duplicates flooded copies,
    // giving generation counts rather than transmission counts.
    std::set<uint64_t> tcSeen;
    uint64_t tcUniqueCount;
    uint64_t tcUniqueRows;
    // Per-source visibility window and inter-arrival statistics over the IPv4
    // frames this vantage point heard. firstSeen/lastSeen give the listener
    // analogue of flow duration; the per-source inter-arrival accumulator gives
    // the listener analogue of per-flow jitter.
    std::map<uint32_t, double> firstSeenBySource;
    std::map<uint32_t, double> lastSeenBySource;
    std::map<uint32_t, double> iatLastBySource;
    std::map<uint32_t, WelfordAcc> iatBySource;
    // Per-hop forwarding latency: an IP packet identity (src,dst,IP-ID) heard
    // again is the next hop's retransmission of the same packet; the delta is
    // the per-hop latency an eavesdropper can actually time. OLSR floods are
    // re-originated with a fresh IP header at every hop, so in practice this
    // times the unicast data path.
    std::map<uint64_t, double> pktFirstHeard;
    WelfordAcc hopDelay;

    void Reset() {
        frames = 0; dataFrames = 0; bytes = 0; tcCount = 0; tcRows = 0;
        helloCount = 0; midCount = 0; hnaCount = 0;
        advByOriginator.clear();
        advTimeByOriginator.clear();
        bytesBySource.clear();
        framesBySource.clear();
        addrsSeen.clear();
        dataBytes = 0; olsrFrames = 0; olsrBytes = 0; tcHopSum = 0;
        tcSeen.clear(); tcUniqueCount = 0; tcUniqueRows = 0;
        firstSeenBySource.clear();
        lastSeenBySource.clear();
        iatLastBySource.clear();
        iatBySource.clear();
        pktFirstHeard.clear();
        hopDelay = WelfordAcc();
    }
    ObserverCounters() { Reset(); }
};
std::vector<ObserverCounters> g_obs;

// ---------------------------------------------------------------------------
// CORRECTED-GLOBAL instrumentation (v4). Air-derivable replacements for the
// metrics whose original collection was found defective. Nothing here alters
// the simulation; every original metric is still written unchanged.
// ---------------------------------------------------------------------------
uint64_t g_phyTxFrames = 0;   // frames whose PHY transmission began: the air truth,
uint64_t g_phyTxBytes  = 0;   // including control frames and retransmissions
// originator -> advertised-link count of its most recent TC, parsed from the
// transmitted packet (TraceOlsrPacket), never from node state. Under an active
// defence this includes the injected fictitious link - that is what the air shows.
std::map<uint32_t, uint32_t> g_lastAdvByOriginator;
// (originator<<16)|msgSeq of every TC seen: distinguishes generated messages
// from flooded copies. Both fields sit in the message header on the air.
std::set<uint64_t> g_tcSeenGlobal;
uint64_t g_tcUniqueCount = 0;
uint64_t g_tcUniqueRows  = 0;
// Every address that appears on the air inside a TC, as the originator or as an
// advertised link. This is the node count as the network SHOWS itself, which is
// what a receiver can know; nodes.GetN() is the simulator's private truth.
//
// Why AverageMprCount needs this as its denominator: the numerator sums
// advertised links, and under an active defence those include the injected
// fictitious node. Dividing that by 50 puts numerator and denominator in
// different populations, so the ratio rises partly because it is mis-scaled and
// not only because the topology changed. The fictitious addresses land in this
// set exactly as real ones do, so they enter both terms and the ratio stays
// self-consistent. §6 of revision_2026_08/STATE.md settled on the same form,
// (sum + F) / (50 + F).
//
// Mirrors the observer's addrsSeen (originator at the TC case, then every
// advertised address), so the global and the single-vantage-point denominators
// are the same quantity measured over different coverage.
std::set<uint32_t> g_addrsSeenGlobal;

// Network-wide ("global") observer: the same ObserverCounters an eavesdropper
// keeps, but fed by every PHY transmission in the network (PhyTxBegin) instead of
// one vantage's receptions. Each transmission is counted once at its transmitter,
// so there is NO "same frame heard by several observers" double counting; a
// relayed packet is counted once per hop, exactly as a single vantage counts the
// relay copies it hears -- consistent air-occupancy coverage, not an artefact.
ObserverCounters g_globalObs;
static void SniffFrameInto (ObserverCounters &oc, Ptr<const Packet> packet);

// ns-3.47 PhyTxBegin signature adds double txPowerW. The count/bytes logic is
// unchanged; txPowerW is ignored, exactly as the 3.19 version ignored nothing.
void PhyTxBeginCallback(std::string context, Ptr<const Packet> packet, double /*txPowerW*/) {
    ++g_phyTxFrames;
    g_phyTxBytes += packet->GetSize();
    // Feed the network-wide observer with the SAME parse code the single-vantage
    // listener uses, so corrected (global) and listener differ only in coverage.
    SniffFrameInto(g_globalObs, packet);
}

// Node id of the isolation attacker chosen at run time. This is a RUN-level
// fact, not a per-window one: it is set when the attack first executes and is
// deliberately never cleared. Making it per-window is not possible here because
// the window-end callback and the attack enable/disable callbacks fire at the
// same simulation instant, so the value would depend on scheduler tie-breaking.
// The observer pool is drawn from the same node range, so a vantage point can
// coincide with the attacker; the id lets such runs be flagged or excluded
// offline.
int32_t g_attackerNodeId = -1;

// Timing constants for the four phases
// ============================================================================
// WINDOW-LENGTH SWEEP (added 13/8/2026, reviewer #3 question 6).
//
// The measurement-window length used to be a hard-coded 40 s. It is now the
// design parameter, settable via --windowSeconds; everything below it is
// DERIVED, so the whole schedule follows from one number:
//
//   MEASUREMENT_DURATION   = windowSeconds
//   UDP_PACKETS_PER_WINDOW = (windowSeconds - UDP_START_OFFSET_IN_WINDOW)
//                            / UDP_PACKET_INTERVAL      (probe rate is FIXED:
//                            one 512-byte datagram every 2 s, starting 4 s
//                            into the window; the datagram count is a result,
//                            not a parameter)
//   window k start         = INITIAL_STABILIZATION
//                            + k * (INTER_STAGE_STABILIZATION + windowSeconds)
//   SIMULATION_END         = DEFENSE_ATTACK_END
//
// With the default --windowSeconds=40 every value below is IDENTICAL to the
// old constants (60/100, 160/200, 260/300, 360/400, 18 packets), so existing
// campaigns reproduce bit-for-bit. windowSeconds must satisfy
// (windowSeconds - 4) % 2 == 0 so the packet count is whole; enforced after
// cmd.Parse in main(), which also calls RecomputeWindowTiming() to rederive
// everything and keeps the actual run long enough (nSimulationSeconds).
//
// These are mutable globals rather than const: every consumer reads them at
// SIMULATION time (Simulator::Schedule callbacks, SaveScenarioMetrics), all
// scheduling happens in main() after cmd.Parse, so no consumer can observe a
// stale value.
// ============================================================================
const double INITIAL_STABILIZATION = 60.0;
const double INTER_STAGE_STABILIZATION = 60.0;
const double UDP_START_OFFSET_IN_WINDOW = 4.0;
// --udpInterval / --udpPacketSize (R#5.7, traffic-density experiment).
// Defaults are the campaign's values, so omitting both flags reproduces
// every earlier run bit-for-bit. UDP_PACKET_INTERVAL is no longer const
// because RecomputeWindowTiming() must see the parsed value.
double   UDP_PACKET_INTERVAL = 2.0;      // --udpInterval, seconds between datagrams
uint32_t g_udpPacketSize     = 512;      // --udpPacketSize, bytes
// --defenseActiveFraction (R#3.11, intermittent-activation experiment).
// Fraction of each DEFENDED measurement window during which the defense is
// active, measured from the window start. Default 1.0 = never deactivated
// mid-window = bit-identical to every earlier campaign.
double   g_defenseActiveFraction = 1.0;

double   g_windowSeconds = 40.0;            // --windowSeconds; the ONE knob
double   MEASUREMENT_DURATION = 40.0;       // derived: = g_windowSeconds
uint32_t UDP_PACKETS_PER_WINDOW = 18;       // derived: (window - offset) / interval

// --enforceHopFilter (default 1 = the paper's behaviour, reject <3 hops).
// 0 = admit every connected run and only RECORD the source-victim hop category.
// Used by the no-hop-filter experiment (STATE §17.14/17.15); the main campaign
// never sets it, so its behaviour is bit-identical to before this flag existed.
bool     g_enforceHopFilter = true;
int      g_hopCategory = 0;                 // 0=undecided, 1/2/3 from AbortOnNeighbor

// --scenarioOrder: the window-order experiment (STATE §23.10, R#4 #1).
//   ""        (default) — the fixed order; the scheduling block is the
//             pre-existing code, byte-identical behaviour.
//   "random"  — per-run permutation of the four phases, derived
//             deterministically from RngRun (printed to stdout; also
//             recoverable from each window's StartTime column).
//   "a,b,c,d" — explicit assignment, e.g. "2,0,3,1" runs phase 2
//             (defense_only) in slot 0, phase 0 (baseline) in slot 1, ...
// Phase indices: 0=baseline 1=attack_only 2=defense_only 3=defense_vs_attack.
// The phases keep their MEANING (label + attack/defense config); only the
// SLOT they occupy changes, so the pipeline sees the same four scenario
// labels in every run regardless of order.
std::string g_scenarioOrder = "";

// --formationLeadIn (seconds, default 0 = the paper's timeline unchanged).
// Prepends a pure network-FORMATION period before the whole four-slot
// schedule, so that slot 0 gets a stabilisation period like every other slot.
//
// Why it exists (STATE §25.20): in the fixed order, slot 0's 60 s is network
// formation, and baseline is the only phase that can live there because it has
// no configuration to apply. Every other slot gets 60 s of settled network
// WITH its config active before its window. Permuting the order breaks that:
// an attack phase landing in slot 0 would have its attacker selected at t=0 on
// empty neighbour tables. With a 60 s lead-in the schedule becomes
//
//   0-60 formation | 60 config | 60-120 stabilise | 120-160 window | ...
//
// so all four slots are structurally identical and every permutation is
// runnable. Both arms of the window-order experiment must use the SAME value.
double g_formationLeadIn = 0.0;

// --propagationModel (STATE §21.4, §23.11, §25.26): the R#2/R#5 realism
// sensitivity experiment.
//   "range"     (default) — the paper's binary 190 m disk, byte-identical.
//   "realistic" — RangePropagationLossModel(--propMaxRange) kept as a
//                 topology GUARD, with LogDistance(--propExponent,
//                 --propRefLoss) + Nakagami fading chained behind it.
// Calibration is via --propRefLoss (NOT via TxPower — §21.4: the 36 dBm the
// FPNT tree needed is 4 W and would invite fresh criticism). The calibration
// target is the SAME mean neighbour degree as the 190 m disk; without it,
// realism and density change together and nothing is attributable.
std::string g_propModel = "range";
double g_propRefLoss  = 46.6777;   // LogDistance default @1m, 5 GHz-ish
double g_propExponent = 3.0;       // urban-ish exponent
double g_propMaxRange = 250.0;     // guard only; fading decides real links

// --rtsCtsThreshold (R#5.3): PSDU size above which the RTS/CTS handshake is
// used. ns-3.47's own default is WIFI_MAX_RTS_THRESHOLD
// (src/wifi/model/wifi-standard-constants.h:113), i.e. RTS/CTS never fires for
// 512-byte datagrams, and this scenario has always relied on that default.
// Hidden-terminal collisions are therefore PRESENT in every campaign this
// project has produced -- they have simply never been isolated. Initialising to
// the same value makes the flag a strict no-op unless it is passed, so every
// existing result stays byte-identical. Set it low (e.g. 100) to suppress
// hidden-terminal collisions and measure their contribution by difference.
uint32_t g_rtsCtsThreshold = 4692480;   // == WIFI_MAX_RTS_THRESHOLD

double BASELINE_START = 60.0;               // derived below
double BASELINE_END = 100.0;

double ATTACK_ONLY_STABILIZATION_START = 100.0;
double ATTACK_ONLY_START = 160.0;
double ATTACK_ONLY_END = 200.0;

double DEFENSE_ONLY_STABILIZATION_START = 200.0;
double DEFENSE_ONLY_START = 260.0;
double DEFENSE_ONLY_END = 300.0;

double DEFENSE_ATTACK_STABILIZATION_START = 300.0;
double DEFENSE_ATTACK_START = 360.0;
double DEFENSE_ATTACK_END = 400.0;

double SIMULATION_END = 400.0;

// Rederive every window boundary from g_windowSeconds. Called once in main()
// immediately after cmd.Parse, BEFORE any Simulator::Schedule call.
static void RecomputeWindowTiming ()
{
  MEASUREMENT_DURATION = g_windowSeconds;
  UDP_PACKETS_PER_WINDOW = (uint32_t) ((g_windowSeconds - UDP_START_OFFSET_IN_WINDOW)
                                       / UDP_PACKET_INTERVAL);

  // g_formationLeadIn (default 0) shifts the ENTIRE schedule later; every
  // boundary below derives from BASELINE_START, and so do the UDP apps and
  // nSimulationSeconds, so this one term moves the whole timeline coherently.
  BASELINE_START = INITIAL_STABILIZATION + g_formationLeadIn;
  BASELINE_END = BASELINE_START + MEASUREMENT_DURATION;

  ATTACK_ONLY_STABILIZATION_START = BASELINE_END;
  ATTACK_ONLY_START = ATTACK_ONLY_STABILIZATION_START + INTER_STAGE_STABILIZATION;
  ATTACK_ONLY_END = ATTACK_ONLY_START + MEASUREMENT_DURATION;

  DEFENSE_ONLY_STABILIZATION_START = ATTACK_ONLY_END;
  DEFENSE_ONLY_START = DEFENSE_ONLY_STABILIZATION_START + INTER_STAGE_STABILIZATION;
  DEFENSE_ONLY_END = DEFENSE_ONLY_START + MEASUREMENT_DURATION;

  DEFENSE_ATTACK_STABILIZATION_START = DEFENSE_ONLY_END;
  DEFENSE_ATTACK_START = DEFENSE_ATTACK_STABILIZATION_START + INTER_STAGE_STABILIZATION;
  DEFENSE_ATTACK_END = DEFENSE_ATTACK_START + MEASUREMENT_DURATION;

  SIMULATION_END = DEFENSE_ATTACK_END;
}

// Base output directory for the four scenarios
std::string g_outputBaseDir = "./simulations/features/";
uint32_t g_currentRun = 1;

// Topology probe for IV-B analysis. Runs once at t=59s for every seed,
// regardless of whether the run is accepted or rejected at t=60s.
// Empty path disables the probe. Path is set via CLI --topologyProbeFile.
std::string g_topologyProbeFile = "";
bool g_probeMobility = false;  // Mirror of bMobility, set after CLI parse

static uint32_t ParseNodeIdFromContext(const std::string& context) {
    // context format: /NodeList/N/DeviceList/...
    std::size_t start = context.find("/NodeList/");
    if (start == std::string::npos) return 0;
    start += 10; // length of "/NodeList/"
    std::size_t end = context.find("/", start);
    if (end == std::string::npos) return 0;
    return static_cast<uint32_t>(std::stoul(context.substr(start, end - start)));
}

void MacTxCallback(std::string context, Ptr<const Packet> packet) {
    uint32_t nodeId = ParseNodeIdFromContext(context);
    if (nodeId < macTxPerNode.size()) {
        macTxPerNode[nodeId]++;
    }
}

void MacTxDropCallback(std::string context, Ptr<const Packet> packet) {
    uint32_t nodeId = ParseNodeIdFromContext(context);
    if (nodeId < macDropPerNode.size()) {
        macDropPerNode[nodeId]++;
    }
}

void
TraceOlsrPacket (Ptr<const Packet> packet, Ptr<Ipv4> ipv4, uint32_t interface)
{
  Ptr<Packet> pktCopy = packet->Copy ();

  Ipv4Header ipHeader;
  pktCopy->RemoveHeader (ipHeader);

  if (ipHeader.GetProtocol () != 17) // Not UDP
    return;

  UdpHeader udpHeader;
  pktCopy->RemoveHeader (udpHeader);

  if (udpHeader.GetDestinationPort () != 698)
    return;
  
  g_olsrControlBytes += packet->GetSize();
  
  // Parse OLSR packet
  iolsr::PacketHeader olsrHeader;
  pktCopy->RemoveHeader (olsrHeader);

  iolsr::MessageHeader msg;

  while (pktCopy->GetSize () > 0)
    {
      if (!pktCopy->RemoveHeader (msg))
        break;

      switch (msg.GetMessageType ())
        {
        case iolsr::MessageHeader::HELLO_MESSAGE:
          ++g_helloCount;
          break;
        case iolsr::MessageHeader::TC_MESSAGE:
          ++g_tcCount;
		  g_totalTcRows += msg.GetTc().neighborAddresses.size();
          // v4 corrected-global: last advertisement per originator, and
          // flood-copy de-duplication by (originator, message sequence number).
          {
            uint32_t adv  = msg.GetTc().neighborAddresses.size();
            uint32_t orig = msg.GetOriginatorAddress().Get();
            g_lastAdvByOriginator[orig] = adv;
            uint64_t key = (static_cast<uint64_t>(orig) << 16)
                         | static_cast<uint64_t>(msg.GetMessageSequenceNumber());
            if (g_tcSeenGlobal.insert(key).second) {
              ++g_tcUniqueCount;
              g_tcUniqueRows += adv;
            }
            // Perceived node set. Counted on every TC, flooded copies included:
            // it is a set, so a repeated address is idempotent, and restricting
            // it to de-duplicated messages would only lose addresses whose
            // originating copy this trace happened not to see first.
            g_addrsSeenGlobal.insert(orig);
            const std::vector<Ipv4Address> &adverts = msg.GetTc().neighborAddresses;
            for (size_t ai = 0; ai < adverts.size(); ++ai) {
              g_addrsSeenGlobal.insert(adverts[ai].Get());
            }
          }
          break;
        case iolsr::MessageHeader::MID_MESSAGE:
          ++g_midCount;
          break;
        case iolsr::MessageHeader::HNA_MESSAGE:
          ++g_hnaCount;
          break;
        default:
          break;
        }
    }
}

// ---------------------------------------------------------------------------
// Passive sniffer callback. Fired only on successful PHY reception, so it
// reproduces exactly what an eavesdropper at that position would capture.
// The packet still carries the full 802.11 stack, hence the manual unwrapping.
// ---------------------------------------------------------------------------
static void SniffFrameInto (ObserverCounters &oc, Ptr<const Packet> packet);

// ns-3.47 MonitorSnifferRx signature: (packet, channelFreqMhz, WifiTxVector,
// MpduInfo, SignalNoiseDbm, staId). Only `packet` is used; the rest are ignored.
void
ObserverSniffRx (uint32_t obsIdx, Ptr<const Packet> packet,
                 uint16_t /*channelFreqMhz*/, WifiTxVector /*txVector*/,
                 MpduInfo /*aMpdu*/, SignalNoiseDbm /*signalNoise*/,
                 uint16_t /*staId*/)
{
  if (obsIdx >= g_obs.size ()) return;
  SniffFrameInto (g_obs[obsIdx], packet);
}

// Shared frame parse: accumulate one frame -- sniffed at a vantage, or (for the
// network-wide observer) transmitted anywhere -- into an ObserverCounters. Both
// callers run identical parse and formula code, so the single-vantage listener
// and the global observer differ only in coverage, never in method.
static void
SniffFrameInto (ObserverCounters &oc, Ptr<const Packet> packet)
{
  oc.frames += 1;
  oc.bytes  += packet->GetSize ();

  Ptr<Packet> pktCopy = packet->Copy ();

  WifiMacHeader macHdr;
  if (pktCopy->GetSize () < macHdr.GetSerializedSize ()) return;
  pktCopy->RemoveHeader (macHdr);
  if (!macHdr.IsData ()) return;          // control/management frames carry no IP

  LlcSnapHeader llc;
  if (pktCopy->GetSize () < llc.GetSerializedSize ()) return;
  pktCopy->RemoveHeader (llc);
  if (llc.GetType () != 0x0800) return;   // not IPv4

  Ipv4Header ipHeader;
  if (pktCopy->GetSize () < ipHeader.GetSerializedSize ()) return;
  pktCopy->RemoveHeader (ipHeader);

  oc.dataFrames += 1;
  oc.dataBytes  += packet->GetSize ();

  // Attribute the frame to its IP source, keyed on the FULL address.
  uint32_t srcKey = ipHeader.GetSource ().Get ();
  oc.bytesBySource[srcKey]  += packet->GetSize ();
  oc.framesBySource[srcKey] += 1;

  // --- v4 timing instrumentation (receive side only) ---
  {
    double nowS = Simulator::Now ().GetSeconds ();
    if (oc.firstSeenBySource.find (srcKey) == oc.firstSeenBySource.end ())
      oc.firstSeenBySource[srcKey] = nowS;
    oc.lastSeenBySource[srcKey] = nowS;
    std::map<uint32_t,double>::iterator li = oc.iatLastBySource.find (srcKey);
    if (li != oc.iatLastBySource.end ())
      oc.iatBySource[srcKey].Add (nowS - li->second);
    oc.iatLastBySource[srcKey] = nowS;

    // Per-hop latency: the same IP identity heard again is the next hop's
    // transmission. Store the latest hearing so successive re-hearings time
    // successive hops rather than the cumulative path. A frame with the 802.11
    // retry bit set is the SAME hop transmitting again after a missed ACK, not
    // the next hop - the bit is on the air, so skipping it stays eavesdropper-
    // derivable (QA finding, 2026-08).
    if (!macHdr.IsRetry ()) {
      uint64_t pk = ((static_cast<uint64_t> (srcKey) << 32)
                     | static_cast<uint64_t> (ipHeader.GetDestination ().Get ()));
      pk = pk * 1000003ULL ^ static_cast<uint64_t> (ipHeader.GetIdentification ());
      std::map<uint64_t,double>::iterator pf = oc.pktFirstHeard.find (pk);
      if (pf == oc.pktFirstHeard.end ()) {
        oc.pktFirstHeard[pk] = nowS;
      } else {
        oc.hopDelay.Add (nowS - pf->second);
        pf->second = nowS;
      }
    }
  }

  if (ipHeader.GetProtocol () != 17) return;   // not UDP

  UdpHeader udpHeader;
  if (pktCopy->GetSize () < udpHeader.GetSerializedSize ()) return;
  pktCopy->RemoveHeader (udpHeader);
  if (udpHeader.GetDestinationPort () != 698) return;   // not OLSR control

  // v4: this frame carries OLSR control - split it out of the byte total so the
  // listener can compute control/data overhead ratios.
  oc.olsrFrames += 1;
  oc.olsrBytes  += packet->GetSize ();

  iolsr::PacketHeader olsrHeader;
  if (pktCopy->GetSize () < olsrHeader.GetSerializedSize ()) return;
  pktCopy->RemoveHeader (olsrHeader);

  // A sniffed 802.11 frame may carry padding past the end of the OLSR packet,
  // so the message loop is bounded by the length field rather than by the
  // remaining buffer size. Feeding padding to MessageHeader::Deserialize would
  // trip its message-type assertion.
  uint32_t olsrRemaining = 0;
  if (olsrHeader.GetPacketLength () > olsrHeader.GetSerializedSize ())
    {
      olsrRemaining = olsrHeader.GetPacketLength () - olsrHeader.GetSerializedSize ();
    }
  if (olsrRemaining > pktCopy->GetSize ()) olsrRemaining = pktCopy->GetSize ();

  iolsr::MessageHeader msg;
  while (olsrRemaining > 0)
    {
      uint32_t before = pktCopy->GetSize ();
      if (!pktCopy->RemoveHeader (msg)) break;
      uint32_t consumed = before - pktCopy->GetSize ();
      if (consumed == 0 || consumed > olsrRemaining) break;
      olsrRemaining -= consumed;

      switch (msg.GetMessageType ())
        {
        case iolsr::MessageHeader::HELLO_MESSAGE:
          ++oc.helloCount;
          break;
        case iolsr::MessageHeader::TC_MESSAGE:
          {
            ++oc.tcCount;
            uint32_t adv = msg.GetTc ().neighborAddresses.size ();
            oc.tcRows += adv;
            uint32_t orig = msg.GetOriginatorAddress ().Get ();
            oc.advByOriginator[orig]     = adv;   // most recent advertisement
            oc.advTimeByOriginator[orig] = Simulator::Now ().GetSeconds ();
            oc.addrsSeen.insert (orig);
            // v4: hop count as carried in the message header, and flood-copy
            // de-duplication so generation counts can be told from re-floods.
            oc.tcHopSum += msg.GetHopCount ();
            {
              uint64_t k2 = (static_cast<uint64_t> (orig) << 16)
                          | static_cast<uint64_t> (msg.GetMessageSequenceNumber ());
              if (oc.tcSeen.insert (k2).second) {
                ++oc.tcUniqueCount;
                oc.tcUniqueRows += adv;
              }
            }
            const std::vector<Ipv4Address> &adverts = msg.GetTc ().neighborAddresses;
            for (size_t ai = 0; ai < adverts.size (); ++ai)
              {
                oc.addrsSeen.insert (adverts[ai].Get ());
              }
          }
          break;
        case iolsr::MessageHeader::MID_MESSAGE:
          ++oc.midCount;
          break;
        case iolsr::MessageHeader::HNA_MESSAGE:
          ++oc.hnaCount;
          break;
        default:
          break;
        }
    }
}

// ---------------------------------------------------------------------------
// Vantage-point selection. Uniform without replacement over the nodes that hold
// no special role (0 = victim, 1 = UDP source, 2 = reserved attacker slot).
// Deterministic in RngRun, so the choice is reproducible and cannot be tuned.
// Any subset of the drawn set is itself a uniform sample of the same size,
// which is what makes the offline K = 1,2,3,5,10 sweep unbiased.
// ---------------------------------------------------------------------------
static void SelectObserverNodes (uint32_t nNodes, uint32_t runSeed)
{
  g_observerNodes.clear ();
  std::vector<uint32_t> pool;
  for (uint32_t i = 3; i < nNodes; ++i) pool.push_back (i);

  // Deterministic Fisher-Yates driven by the run seed.
  uint64_t state = 6364136223846793005ULL * (uint64_t) runSeed + 1442695040888963407ULL;
  for (uint32_t i = (uint32_t) pool.size (); i > 1; --i)
    {
      state = 6364136223846793005ULL * state + 1442695040888963407ULL;
      uint32_t j = (uint32_t) ((state >> 33) % i);
      std::swap (pool[i - 1], pool[j]);
    }

  uint32_t k = NUM_OBSERVERS < pool.size () ? NUM_OBSERVERS : (uint32_t) pool.size ();
  for (uint32_t i = 0; i < k; ++i) g_observerNodes.push_back (pool[i]);

  g_obs.assign (g_observerNodes.size (), ObserverCounters ());
}
 
static void PrintCountFakeNodes(NodeContainer* cont){
/*
Count the number of nodes that require fake/fictive neighbor
A fake node is required when there is a risk that some node will execute an isolation attack
*/
	//ns3::iolsr::RoutingProtocol rp;
	unsigned int count = 0;
	std::cout << "Nodes who are required to advertise fictive: " ;
	for (unsigned int i=0;i<cont->GetN();++i){
		Ptr<RoutingProtocol> pt = cont->Get(i)->GetObject<RoutingProtocol>();
		//NS_LOG_INFO("Node id: " << i << "\tTime: " << Simulator::Now().GetSeconds() << "\tRequireFake? " << pt->RequireFake());
		if (pt->RequireFake()){
			++count;
			std::cout << i << ", ";
		} ;
	}
	std::cout << std::endl;
	NS_LOG_INFO("Total fakes required: " << count << " out of " << cont->GetN());
	std::cout << "Fictives req.: " << count << std::endl;
}

static void PrintNodesDeclaredFictive(NodeContainer* cont){

	//ns3::iolsr::RoutingProtocol rp;
	unsigned int count = 0;
	std::cout << "Nodes who declared fictive: ";
	for (unsigned int i=0;i<cont->GetN();++i){
		Ptr<RoutingProtocol> pt = cont->Get(i)->GetObject<RoutingProtocol>();
		//NS_LOG_INFO("Node id: " << i << "\tTime: " << Simulator::Now().GetSeconds() << "\tRequireFake? " << pt->RequireFake());
		
		if (pt->returnDeclaredFictive()){
			++count;
			std::cout << i << ", ";
		} 
	}
	std::cout << std::endl;
	NS_LOG_INFO("Total nodes declaring fictives: " << count << " out of " << cont->GetN());
	std::cout << "Total nodes declaring fictives: " << count << std::endl;
}

static void PrintMprFraction(NodeContainer* cont){
/*
For each node get the fraction of it's MPR (using FractionOfMpr: number of MPRs / number of neighbors) and sum all the fractions
Output a message of the fraction of MPR over all nodes
*/
	// double sum = 0;
	// for (unsigned int i=0;i<cont->GetN();++i){
	// 	Ptr<RoutingProtocol> pt = cont->Get(i)->GetObject<RoutingProtocol>();
	// 	//NS_LOG_INFO("Node id: " << i << "\tTime: " << Simulator::Now().GetSeconds() << "\tFraction: " << pt->FractionOfMpr());
	// 	sum += pt->FractionOfMpr();
	// }
	// NS_LOG_INFO("Total fraction of MPR: " << sum / (double) cont->GetN());
	// NS_LOG_INFO(sum / (double) cont->GetN());
	// //... NS_LOG_INFO broke. Using cout...
	// //std::cout << "Total fraction of MPR: " << sum / (double) cont->GetN() << std::endl;
	// std::cout << "MPR: " << (sum / (double) cont->GetN()) << std::endl; 

	double sum = 0;
	uint32_t countTotalNeighbors = 0;
	for (unsigned int i=0;i<cont->GetN();++i){
		Ptr<RoutingProtocol> pt = cont->Get(i)->GetObject<RoutingProtocol>();
		//NS_LOG_INFO("Node id: " << i << "\tTime: " << Simulator::Now().GetSeconds() << "\tFraction: " << pt->FractionOfMpr());
		sum += pt->getMprSize();
		countTotalNeighbors  += pt->getNeighborsSize();
	}
	// NS_LOG_INFO("Total fraction of MPR: " << sum / (double) cont->GetN());
	// NS_LOG_INFO(sum / (double) cont->GetN());
	NS_LOG_INFO("Total fraction of MPR: " << sum / (double) countTotalNeighbors);
	NS_LOG_INFO(sum / (double) countTotalNeighbors);
	//... NS_LOG_INFO broke. Using cout...
	//std::cout << "Total fraction of MPR: " << sum / (double) cont->GetN() << std::endl;
	std::cout << "Total MPRs chosen: " << sum << std::endl;
	std::cout << "Total Neighbors: " << countTotalNeighbors << std::endl;

	std::cout << "Avg MPR fraction: " << (sum / countTotalNeighbors) << std::endl;
}

static void PrintMprs(NodeContainer* cont){
	uint32_t totalMprs = 0;
	MprSet allMPRs = cont->Get(0)->GetObject<RoutingProtocol>()->getMprSet();
	for (unsigned int i=1;i<cont->GetN();++i){
		MprSet currentMprSet = cont->Get(i)->GetObject<RoutingProtocol>()->getMprSet();
		allMPRs.insert(currentMprSet.begin(), currentMprSet.end());
		
	// 	sumLsr += pt->tcPowerLevel(true);
	// 	sumOlsr += pt->tcPowerLevel(false);
	// }
	// //NS_LOG_INFO("TC Level: " << sumOlsr / (double) cont->GetN() << " (lsr: " << sumLsr / (double) cont->GetN() << ")");
	// std::cout << "TC Level: " << sumOlsr / (double) cont->GetN() << " \nlsr: " << sumLsr / (double) cont->GetN() << "\n";
	}
	totalMprs = allMPRs.size();
	std::cout << "Total MPRs: " << totalMprs << std::endl;
	std::cout << "MPR sub-network nodes: " << std::endl;
	for (MprSet::const_iterator it = allMPRs.begin(); it != allMPRs.end(); ++it){
			std::cout << *it << ", ";
		}
	std::cout << std::endl;
}

static void Print2hopNeighborsOfVictim(NodeContainer* cont){
	Ptr<RoutingProtocol> victimNode = cont->Get(0)->GetObject<RoutingProtocol>();

	const NeighborSet& neighbors = victimNode->getNeighborSet();
	const TwoHopNeighborSet& twoHops = victimNode->getTwoHopNeighborSet();
	std::list<Ipv4Address> oneHopsInList;
	std::list<Ipv4Address> twoHopsInList;
	for (NeighborSet::const_iterator it = neighbors.begin(); it != neighbors.end(); ++it){
		//if (it->twoHopNeighborAddr == m_mainAddress) continue;
      //std::cout << it->twoHopNeighborAddr << ", " ;
	  oneHopsInList.push_back(it->neighborMainAddr);
    }
	for (TwoHopNeighborSet::const_iterator it = twoHops.begin(); it != twoHops.end(); ++it){
		//if (it->twoHopNeighborAddr == m_mainAddress) continue;
      //std::cout << it->twoHopNeighborAddr << ", " ;
	  twoHopsInList.push_back(it->twoHopNeighborAddr);
    }
std::cout << "1Hop from victim: " << std::endl;
	oneHopsInList.sort();
	oneHopsInList.unique();
	for (std::list<Ipv4Address>::iterator it = oneHopsInList.begin(); it != oneHopsInList.end(); ++it){
		std::cout << *it << ", " ;
	}
	std::cout << std::endl;

	std::cout << "2Hops from victim: " << std::endl;
	twoHopsInList.sort();
	twoHopsInList.unique();
	for (std::list<Ipv4Address>::iterator it = twoHopsInList.begin(); it != twoHopsInList.end(); ++it){
		std::cout << *it << ", " ;
	}
	std::cout << std::endl;
}

static void PrintC6Detection(NodeContainer* cont){
/*
Count the number of nodes that detected themselves as part of a c6 cycle topology.
*/
	//ns3::iolsr::RoutingProtocol rp;
	unsigned int count = 0;
	std::cout << "Nodes detected as part of c6: ";
	for (unsigned int i=0;i<cont->GetN();++i){
		Ptr<RoutingProtocol> pt = cont->Get(i)->GetObject<RoutingProtocol>();
		//NS_LOG_INFO("Node id: " << i << "\tTime: " << Simulator::Now().GetSeconds() << "\tRequireFake? " << pt->RequireFake());
		if (pt->returnDetectedInC6()){
			std::cout << i << ", ";
			++count;
		}
	}
	NS_LOG_INFO("Total nodes in c6: " << count << " out of " << cont->GetN());
	std::cout << std::endl;
	std::cout << "Num. of nodes detected as part of c6: " << count << std::endl;
}

static void PrintNodeOutputLog(NodeContainer* cont, uint32_t nodeID){ //!!
	Ptr<RoutingProtocol> pt = cont->Get(nodeID)->GetObject<RoutingProtocol>();
	std::string log = pt->getOutputLog();
	std::cout << "Log for node " << nodeID << ": " << std::endl;
	std::cout << log << std::endl << std::endl;
}

static void PrintRiskyFraction(NodeContainer* cont, Ipv4Address ignore = Ipv4Address("0.0.0.0")){
	/*
	*/
	double sum = 0;
	for (unsigned int i=0;i<cont->GetN();++i){
		Ptr<RoutingProtocol> pt = cont->Get(i)->GetObject<RoutingProtocol>();
		sum += pt->FractionOfNodesMarkedAsRisky(ignore);
	}
	std::cout << "Risky nodes: " << sum << std::endl;
}

static void PrintTcPowerLevel(NodeContainer* cont){
	double sumLsr = 0;
	double sumOlsr = 0;
	for (unsigned int i=0;i<cont->GetN();++i){
		Ptr<RoutingProtocol> pt = cont->Get(i)->GetObject<RoutingProtocol>();
		sumLsr += pt->tcPowerLevel(true);
		sumOlsr += pt->tcPowerLevel(false);
	}
	//NS_LOG_INFO("TC Level: " << sumOlsr / (double) cont->GetN() << " (lsr: " << sumLsr / (double) cont->GetN() << ")");
	std::cout << "TC Level: " << sumOlsr / (double) cont->GetN() << " \nlsr: " << sumLsr / (double) cont->GetN() << "\n";
}

static double getTcPowerLevel(NodeContainer* cont){
	double sumOlsr = 0;
	for (unsigned int i=0;i<cont->GetN();++i){
		Ptr<RoutingProtocol> pt = cont->Get(i)->GetObject<RoutingProtocol>();
		sumOlsr += pt->tcPowerLevel(false);
	}
	return sumOlsr / (double) cont->GetN();
}

static void ExecuteIsolationAttack(NodeContainer* cont){
	// Make it pick a node at random later...
	Ipv4Address target = cont->Get(2)->GetObject<RoutingProtocol>()->ExecuteIsolationAttack();
	target.IsBroadcast(); // Kill the unused warning
	//NS_LOG_INFO("Executing node isolation attack on: " << target);
	//NS_LOG_INFO("Attacker address: " << cont->Get(2)->GetObject<Ipv4>()->GetAddress(1,0));
	//NS_LOG_INFO("Victim  test: " << cont->Get(25)->GetObject<Ipv4>()->GetAddress(1,0));
	std::cout << "Executing node isolation attack on: " << target << std::endl;
	std::cout << "Attacker address: " << cont->Get(2)->GetObject<Ipv4>()->GetAddress(1,0) << std::endl;
}

static void ExecuteIsolationAttackMassive(NodeContainer* cont){
	// Let the first 30% nodes attack and see what happens
	for (uint32_t i=5;i<cont->GetN() * 0.3 + 5; ++i){
		// cont->Get(i)->GetObject<RoutingProtocol>()->ExecuteIsolationAttack();
		cont->Get(i)->GetObject<RoutingProtocol>()->ExecuteIsolationAttack(Ipv4Address("10.0.0.1"));
	}
}

static void ExecuteIsolationAttackByNeighbor(NodeContainer* nodes, Ipv4Address target){
	bool found = false;
	for (uint32_t i=3; i< nodes->GetN(); ++i){
		if (nodes->Get(i)->GetObject<RoutingProtocol>()->isItNeighbor(target)){
			found = true;
			g_attackerNodeId = static_cast<int32_t>(i);
			nodes->Get(i)->GetObject<RoutingProtocol>()->ExecuteIsolationAttack(target);
			std::cout << "Attacker found. Attacking from node id: " << i << std::endl;
			break;
		}
	}
	if(!found){
		std::cout << "Attacker not available. *** Terminated *** " << std::endl;
		Simulator::Stop();
	}
}

// Mirror of ExecuteIsolationAttackByNeighbor for the ported student-DCFM
// black-hole attack: pick a real neighbor of the target and turn it into a
// black-hole attacker. Sets the per-node IsMalicious + SpoofedLinksCount
// attributes (registered in iolsr-routing-protocol.cc), which enable HELLO/TC
// real-target spoofing, ANSN poisoning, and forwarded-data dropping. Every
// other node keeps IsMalicious=false (the default) and stays benign.
static void ExecuteBlackholeAttackByNeighbor(NodeContainer* nodes, Ipv4Address target, uint32_t spoofCount){
	bool found = false;
	for (uint32_t i=3; i< nodes->GetN(); ++i){
		if (nodes->Get(i)->GetObject<RoutingProtocol>()->isItNeighbor(target)){
			found = true;
			Ptr<RoutingProtocol> rp = nodes->Get(i)->GetObject<RoutingProtocol>();
			rp->SetAttribute("IsMalicious", BooleanValue(true));
			rp->SetAttribute("SpoofedLinksCount", UintegerValue(spoofCount));
			std::cout << "Black-hole attacker found. Attacking from node id: " << i
			          << " (spoofedLinks=" << spoofCount << ")" << std::endl;
			break;
		}
	}
	if(!found){
		std::cout << "Attacker not available. *** Terminated *** " << std::endl;
		Simulator::Stop();
	}
}

// Aligned with the student DCFM harness: activate the black-hole on a FIXED node
// id (default 2), which the mobility setup pins at the grid centre so it has high
// betweenness and actually sits on data paths. Replaces the neighbor-of-victim
// selection, which left the attacker off the single flow path.
static void ExecuteBlackholeAttackOnNode(NodeContainer* nodes, uint32_t nodeId, uint32_t spoofCount){
	if (nodeId >= nodes->GetN()){
		std::cout << "Black-hole attacker node id " << nodeId << " out of range. *** Terminated *** " << std::endl;
		Simulator::Stop();
		return;
	}
	Ptr<RoutingProtocol> rp = nodes->Get(nodeId)->GetObject<RoutingProtocol>();
	rp->SetAttribute("IsMalicious", BooleanValue(true));
	rp->SetAttribute("SpoofedLinksCount", UintegerValue(spoofCount));
	std::cout << "Black-hole attacker = node id: " << nodeId
	          << " (central placement, spoofedLinks=" << spoofCount << ")" << std::endl;
}

// Clear the black-hole state on every node (mirror of DisableIsolationAttack),
// used at phase boundaries so the attack is cleanly off outside attack windows.
static void DisableBlackholeAttack(NodeContainer* cont){
	for (unsigned int i = 0; i < cont->GetN(); ++i){
		Ptr<RoutingProtocol> rp = cont->Get(i)->GetObject<RoutingProtocol>();
		rp->SetAttribute("IsMalicious", BooleanValue(false));
		rp->SetAttribute("SpoofedLinksCount", UintegerValue(0));
	}
}

static void PercentageWithFullConnectivity(NodeContainer* cont){
	uint32_t count = 0;
	for (uint32_t i=0;i<cont->GetN();++i){
		uint32_t routingTableSize = cont->Get(i)->GetObject<RoutingProtocol>()->getRoutingTableSize();
		MprSet nodeMprSet = cont->Get(i)->GetObject<RoutingProtocol>()->getMprSet();
		
		std::cout << "Node id: "<< i << ", nodes in routing table: " << routingTableSize << ".   MPRs(" << nodeMprSet.size() <<"): ";
		for (MprSet::const_iterator it = nodeMprSet.begin(); it != nodeMprSet.end(); ++it){
			std::cout << *it << ", ";
		}
		std::cout << std::endl;
		if (routingTableSize == cont->GetN() - 1) {
			++count;
		}
	}
	double result = count / (double) cont->GetN();
	std::cout << "Routing Percentage: " << result << "\n";
	//Simulator::Schedule(Seconds (10), &PercentageWithFullConnectivity, cont);
}

static void ActivateFictiveDefence(NodeContainer* cont){
	for (unsigned int i=0; i < cont->GetN(); ++i){
		cont->Get(i)->GetObject<RoutingProtocol>()->activateFictiveDefence();
	}
}

static void ActivateFictiveMitigation(NodeContainer* cont){
	for (unsigned int i=0; i < cont->GetN(); ++i){
		cont->Get(i)->GetObject<RoutingProtocol>()->activateFictiveMitigation();
	}
	//std::cout << "Enabled new mitigation defence on all nodes." << std::endl;
}

static void DisableIsolationAttack(NodeContainer* cont){
    for (unsigned int i = 0; i < cont->GetN(); ++i){
        cont->Get(i)->GetObject<RoutingProtocol>()->disableIsolationAttack();
    }
}

static void DeactivateFictiveDefence(NodeContainer* cont){
    for (unsigned int i = 0; i < cont->GetN(); ++i){
        cont->Get(i)->GetObject<RoutingProtocol>()->deactivateFictiveDefence();
    }
}

static void DeactivateFictiveMitigation(NodeContainer* cont){
    for (unsigned int i = 0; i < cont->GetN(); ++i){
        cont->Get(i)->GetObject<RoutingProtocol>()->deactivateFictiveMitigation();
    }
}

static void ReportNumReceivedPackets(Ptr<UdpServer> udpServer){
	std::cout << "Packets: " << udpServer->GetReceived() << std::endl;
}

static void AbortIfNotReceivedPackets(Ptr<UdpServer> udpServer){
    if (!udpServer) return;
	uint32_t receivedInWindow = udpServer->GetReceived() - g_udpReceivedAtWindowStart;
    if (receivedInWindow == 0){
        std::cout << "* Received 0 packets in window, terminating." << std::endl;
        Simulator::Stop();
    }
}

static void SnapshotUdpReceived(Ptr<UdpServer> udpServer) {
    if (!udpServer) return;
	g_udpReceivedAtWindowStart = udpServer->GetReceived();
}

static void AssertConnectivity(NodeContainer* cont){
	for (unsigned int i=0; i < cont->GetN(); ++i){
		if (cont->Get(i)->GetObject<RoutingProtocol>()->getRoutingTableSize() != cont->GetN() - 1) {
			std::cout << "*** Assert connectivity failed, terminated." << std::endl;
			Simulator::Stop();
		}
	}
	//if (cont->Get(0)->GetObject<RoutingProtocol>()->getRoutingTableSize() != cont->GetN() - 1) {
	//	Simulator::Stop();
	//}
}

// Scenario-viability admission for the realistic-channel experiment
// (STATE §25.36). AssertConnectivity above requires EVERY node's table to be
// complete at t=60 — an all-or-nothing snapshot that is unmeetable under
// Nakagami fading (some link always drops a HELLO), which rejected 2000/2000
// realistic-channel seeds. It was a PROXY for "the scenario can actually run";
// this checks the two conditions that proxy stood for, directly:
//   1. the victim (node 0, 10.0.0.1) has a route to the UDP source (node 1),
//      so the traffic the windows measure will actually flow;
//   2. the victim has at least one neighbour, so an attacker can be selected
//      and the isolation attack is exercised.
// Selecting on these — rather than on full-table completeness — keeps the
// population defined by scenario feasibility, not by channel luck, and the
// rejection rate is itself reported as a finding. Gated by
// --admission=scenario; the default 'full' path is the paper's, byte-identical.
static void AssertScenarioViable(NodeContainer* cont){
	if (cont->GetN() < 2) { Simulator::Stop(); return; }
	Ptr<RoutingProtocol> victim = cont->Get(0)->GetObject<RoutingProtocol>();
	// (1) a route from the victim to the source node (node 1 == 10.0.0.2)
	if (!victim->HasRouteTo(Ipv4Address("10.0.0.2"))) {
		std::cout << "*** Scenario not viable: victim has no route to source, terminated." << std::endl;
		Simulator::Stop();
		return;
	}
	// (2) the victim has at least one one-hop neighbour (an attacker exists)
	if (victim->getNeighborSet().empty()) {
		std::cout << "*** Scenario not viable: victim has no neighbour, terminated." << std::endl;
		Simulator::Stop();
		return;
	}
}
static void AbortOnNeighbor (Ptr<Node> node, Ipv4Address address){
	// Acceptance criterion, evaluated at the start of every measurement window.
	//
	// The paper's criterion requires the UDP source to be at least THREE hops
	// from the victim. OLSR builds routing-table entries for one- and two-hop
	// destinations from the HELLO-derived neighbour and two-hop neighbour sets,
	// and only entries at three hops and beyond from the TC-derived topology
	// set -- which is what the isolation attack poisons. A closer source
	// therefore never exercises the attack.
	//
	// --enforceHopFilter=0 keeps such runs instead of discarding them, and
	// records how close the source actually was:
	//     1 = direct neighbour, 2 = two hops, 3 = three or more (the paper's
	//     admitted population).
	//
	// The category is printed to STDOUT ONLY. It must NEVER reach
	// metrics_output: make_arm_bundles.py turns every metric row into a feature
	// column, and hop distance correlates with how easy the run is to classify,
	// so as a feature it would leak. The campaign script captures this line into
	// its own metadata file, keyed by run id, and any per-distance breakdown is
	// computed AFTER training by joining on file_source.
	Ptr<RoutingProtocol> rp = node->GetObject<RoutingProtocol>();
	int cat = 3;
	if (rp->isItNeighbor(address))          cat = 1;
	else if (rp->isItUpToTwoHop(address))   cat = 2;

	// Record per WINDOW and keep the MINIMUM, not the first value.
	//
	// Under mobility the nodes move between windows, so the source-victim
	// distance is not a property of the run: it can be four hops at the baseline
	// window and one hop at the last. Taking the first window would mislabel
	// those runs.
	//
	// The minimum reproduces the original filter's decision exactly: the paper's
	// criterion is evaluated at the start of every window and Simulator::Stop()
	// fires on the FIRST violation, so a run survives only if it was >=3 hops at
	// all four checks -- i.e. iff its minimum category is 3.
	//
	// HOP_WINDOW is one line per window; HOP_CATEGORY is the running minimum, so
	// the LAST HOP_CATEGORY line of a run is its final classification. Both go to
	// stdout only, never into metrics_output (see the leak note above).
	std::cout << "HOP_WINDOW=" << cat << std::endl;
	if (g_hopCategory == 0 || cat < g_hopCategory) {
		g_hopCategory = cat;
	}
	std::cout << "HOP_CATEGORY=" << g_hopCategory << std::endl;

	if (g_enforceHopFilter && cat < 3){
		// Message text is unchanged: run_campaign_v347.sh greps for it.
		std::cout << "*** Sending node of udp packets is within two hops of victim (needs >=3). Terminated." << std::endl;
		Simulator::Stop();
	}
}

static void TrackTarget (Ptr<Node> target, Ptr<Node> tracker){
	Vector vec = target->GetObject<MobilityModel>()->GetPosition();
	vec.x += 8;
	vec.y += 0;
	tracker->GetObject<MobilityModel>()->SetPosition(vec);
}

static void PrintTables(Ptr<Node> n, std::string fname){
	std::ofstream o;
	o.open((std::string("_TwoHop") + fname + std::string(".txt")).c_str());
	const TwoHopNeighborSet &two = n->GetObject<RoutingProtocol>()->getTwoHopNeighborSet();
	for (TwoHopNeighborSet::const_iterator it = two.begin(); it!=two.end(); ++it){
		o << *it << "\n";
	}
	o.close();
	o.open((std::string("_Topology") + fname + std::string(".txt")).c_str());
	const TopologySet &tp = n->GetObject<RoutingProtocol>()->getTopologySet();
	for (TopologySet::const_iterator it = tp.begin(); it!=tp.end(); ++it){
		o << *it << "\n";
	}
	o.close();
	o.open((std::string("_Neighbor") + fname + std::string(".txt")).c_str());
	const NeighborSet &nei = n->GetObject<RoutingProtocol>()->getNeighborSet();
	for (NeighborSet::const_iterator it = nei.begin(); it!=nei.end(); ++it){
		o << *it << "\n";
	}
	o.close();
}

static void IneffectiveNeighorWrite(NodeContainer *cont){
	std::ofstream o;
	o.open ("_mpr.txt");
	//For each node write a list of it's MPRs
	for (unsigned int i=0; i< cont->GetN(); ++i){
		const MprSet mpr = cont->Get(i)->GetObject<RoutingProtocol>()->getMprSet();
		o << cont->Get(i)->GetObject<Ipv4>()->GetAddress(1,0).GetLocal() << ":";
		for (MprSet::const_iterator it = mpr.begin(); it != mpr.end(); ++it){
			o << *it << ",";
		}
		o << "\n";
	}
	o.close();
	o.open ("_neighbors.txt");
	//For each node write a list of it's 1-hop neighbors and their address
	for (unsigned int i=0; i< cont->GetN(); ++i){
		const NeighborSet neighbor = cont->Get(i)->GetObject<RoutingProtocol>()->getNeighborSet();
		o << cont->Get(i)->GetObject<Ipv4>()->GetAddress(1,0).GetLocal() << ":";
		for (NeighborSet::const_iterator it = neighbor.begin(); it != neighbor.end(); ++it){
			o << it->neighborMainAddr << ",";
		}
		o << "\n";
	}
	o.close();

}

bool bIsolationAttackBug;
bool bEnableFictive;
bool bIsolationAttackNeighbor;
bool bEnableFictiveMitigation;

// static void PrintSimStats(NodeContainer* cont){

// 	std::cout << "@@   New Simulation   @@" << std::endl;
// 	std::cout << "Seed RngRun: " << RngSeedManager::GetRun() << std::endl;
// 	if(bIsolationAttackBug || bIsolationAttackNeighbor){
// 	std::cout << "Attack: ON" << std::endl;
// 	} else{
// 		std::cout << "Attack: OFF" << std::endl;
// 	}
// 	if(bEnableFictive || bEnableFictiveMitigation){
// 		std::cout << "Defence: ON" << std::endl;
// 	} else{
// 		std::cout << "Defence: OFF" << std::endl;
// 	}

	
// }

// Deprecated: legacy metric extraction path.
// Not used in the four-window methodology.
void ExtractAndLogMetrics(Ptr<FlowMonitor> flowMon, FlowMonitorHelper &flowHelper, NodeContainer &nodes, const char* filename) {
    std::ofstream file(filename);
    file << "Metric,Value" << std::endl;

    flowMon->CheckForLostPackets();
    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(flowHelper.GetClassifier());
    std::map<FlowId, FlowMonitor::FlowStats> stats = flowMon->GetFlowStats();

    double totalTxPackets = 0, totalRxPackets = 0, totalLostPackets = 0, totalDelay = 0;
    double totalThroughput = 0, totalJitter = 0, totalHopCount = 0, totalRoutingPackets = 0;
    double totalEnergyConsumed = 0;

    std::map<FlowId, FlowMonitor::FlowStats>::iterator it;
    for (it = stats.begin(); it != stats.end(); ++it) {
        totalTxPackets += it->second.txPackets;
        totalRxPackets += it->second.rxPackets;
        totalLostPackets += (it->second.txPackets - it->second.rxPackets);
        totalDelay += it->second.delaySum.GetSeconds();
        totalJitter += it->second.jitterSum.GetSeconds();
        totalHopCount += (it->second.timesForwarded + 1);
        totalThroughput += it->second.rxBytes * 8.0 / (it->second.timeLastRxPacket.GetSeconds() - it->second.timeFirstTxPacket.GetSeconds()) / 1e6;
    }

    double pdr = (totalTxPackets > 0) ? (totalRxPackets / totalTxPackets) * 100 : 0;
    double plr = (totalTxPackets > 0) ? (totalLostPackets / totalTxPackets) * 100 : 0;
    double avgDelay = (totalRxPackets > 0) ? (totalDelay / totalRxPackets) : 0;
    double avgJitter = (totalRxPackets > 0) ? (totalJitter / totalRxPackets) : 0;
    double avgHopCount = (totalRxPackets > 0) ? (totalHopCount / totalRxPackets) : 0;

    NodeContainer::Iterator i;
    for (i = nodes.Begin(); i != nodes.End(); ++i) {
        Ptr<iolsr::RoutingProtocol> iolsr = (*i)->GetObject<iolsr::RoutingProtocol>();
        if (iolsr) {
            totalRoutingPackets += iolsr->getRoutingTableSize();
        }

        Ptr<energy::EnergySource> energySource = (*i)->GetObject<energy::EnergySource>();  // ns-3.47: energy:: namespace
        if (energySource) {
		totalEnergyConsumed += (energySource->GetInitialEnergy() - energySource->GetRemainingEnergy());
        }
    }

    double avgSpeed = 0;
    for (i = nodes.Begin(); i != nodes.End(); ++i) {
        Ptr<MobilityModel> mobilityModel = (*i)->GetObject<MobilityModel>();
        Vector velocity = mobilityModel->GetVelocity();
        avgSpeed += std::sqrt(std::pow(velocity.x, 2) + std::pow(velocity.y, 2) + std::pow(velocity.z, 2));
    }
    avgSpeed /= nodes.GetN();

    double energyEfficiency = (totalEnergyConsumed > 0) ? totalThroughput / totalEnergyConsumed : 0;
    double normalizedRoutingLoad = (totalRxPackets > 0) ? totalRoutingPackets / totalRxPackets : 0;
    double avgTcRows = getTcPowerLevel(&nodes);
    double routingOverhead = (totalRxPackets > 0) ? totalRoutingPackets / totalRxPackets : 0;

    file << "Packet Delivery Ratio (%)," << pdr << std::endl;
    file << "Packet Loss Ratio (%)," << plr << std::endl;
    file << "End-to-End Delay (s)," << avgDelay << std::endl;
    file << "Jitter (s)," << avgJitter << std::endl;
    file << "Throughput (Mbps)," << totalThroughput << std::endl;
    file << "Average Hop Count," << avgHopCount << std::endl;
    file << "Total Energy Consumed (J)," << totalEnergyConsumed << std::endl;
    file << "Average Node Speed (m/s)," << avgSpeed << std::endl;
    file << "Energy Efficiency (bits/Joule)," << energyEfficiency << std::endl;
    file << "Normalized Routing Load," << normalizedRoutingLoad << std::endl;
    file << "Average TC Packet Rows," << avgTcRows << std::endl;
    file << "Routing Overhead," << routingOverhead << std::endl;
    file << "HELLO packets," << g_helloCount / nodes.GetN() << std::endl;
    file << "TC packets," << g_tcCount / nodes.GetN() << std::endl;
    file << "MID packets," << g_midCount / nodes.GetN() << std::endl;
    file << "HNA packets," << g_hnaCount / nodes.GetN() << std::endl;

    file.close();
    std::cout << "Metrics written to " << filename << std::endl;
}

// Function to create output directories
static void CreateOutputDirectories() {
    system(("mkdir -p " + g_outputBaseDir + "baseline/").c_str());
	system(("mkdir -p " + g_outputBaseDir + "attack_only/").c_str());
    system(("mkdir -p " + g_outputBaseDir + "defense_only/").c_str());
    system(("mkdir -p " + g_outputBaseDir + "defense_vs_attack/").c_str());
}

// Convert number to string (C++11 compatible)
std::string ToString(uint32_t value) {
    std::ostringstream oss;
    oss << value;
    return oss.str();
}

std::string ToString(double value) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6) << value;
    return oss.str();
}

static double ComputeMean(const std::vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }

    double sum = 0.0;
    for (std::vector<double>::const_iterator it = values.begin(); it != values.end(); ++it) {
        sum += *it;
    }
    return sum / static_cast<double>(values.size());
}

static double ComputeStdDev(const std::vector<double>& values, double mean) {
    if (values.empty()) {
        return 0.0;
    }

    double sqSum = 0.0;
    for (std::vector<double>::const_iterator it = values.begin(); it != values.end(); ++it) {
        double diff = (*it - mean);
        sqSum += diff * diff;
    }
    return std::sqrt(sqSum / static_cast<double>(values.size()));
}

static void WriteScenarioMetric(std::ofstream& file,
                                const std::string& scenario,
                                const std::string& metric,
                                double value,
                                double startTime,
                                double endTime,
                                double measurementDuration) {
    file << scenario << "," << metric << "," << ToString(value) << ","
         << ToString(startTime) << "," << ToString(endTime) << ","
         << ToString(measurementDuration) << "\n";
}


// Function to save metrics for a specific scenario (C++11 compatible with all metrics)
static void SaveScenarioMetrics(const std::string& scenario, NodeContainer& nodes, 
                               double startTime, double endTime) {
    
    std::string filename = g_outputBaseDir + scenario + "/metrics_output-" + 
                          ToString(g_currentRun) + ".csv";
    
    std::ofstream file(filename.c_str());
    if (!file.is_open()) {
        NS_LOG_ERROR("Cannot open file: " << filename);
        return;
    }
    
    // Write CSV header
    file << "Scenario,Metric,Value,StartTime,EndTime,Duration\n";
    file << std::fixed << std::setprecision(6);
    
    // Calculate metrics using your existing ExtractAndLogMetrics logic
	if (g_flowMonitor) {
        g_flowMonitor->CheckForLostPackets();
        Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(g_flowHelper.GetClassifier());
        std::map<FlowId, FlowMonitor::FlowStats> stats = g_flowMonitor->GetFlowStats();

        double totalTxPackets = 0.0, totalRxPackets = 0.0, totalLostPackets = 0.0;
		double totalTxBytes = 0.0, totalRxBytes = 0.0;
		double totalDelay = 0.0, totalJitter = 0.0, totalForwardings = 0.0;
		double measurementDuration = endTime - startTime;

		std::vector<double> flowDurations;
		std::vector<double> flowThroughputs;
		std::vector<double> flowAvgDelays;
		std::vector<double> flowAvgJitters;
		std::vector<double> flowLossRates;
		uint32_t flowCount = 0;

		// C++11 compatible iteration
		std::map<FlowId, FlowMonitor::FlowStats>::iterator it;
for (it = stats.begin(); it != stats.end(); ++it) {
    const FlowMonitor::FlowStats& fs = it->second;

    bool isActiveFlow =
        (fs.txPackets > 0) || (fs.rxPackets > 0) || (fs.txBytes > 0) || (fs.rxBytes > 0);

    if (!isActiveFlow) {
        continue;
    }

    ++flowCount;

    totalTxPackets += fs.txPackets;
    totalRxPackets += fs.rxPackets;
    totalLostPackets += (fs.txPackets - fs.rxPackets);
    totalTxBytes += fs.txBytes;
    totalRxBytes += fs.rxBytes;
    totalDelay += fs.delaySum.GetSeconds();
    totalJitter += fs.jitterSum.GetSeconds();
    totalForwardings += fs.timesForwarded;

    double firstTx = fs.timeFirstTxPacket.GetSeconds();
    double lastTx = fs.timeLastTxPacket.GetSeconds();
    double lastRx = fs.timeLastRxPacket.GetSeconds();
    double flowEnd = (lastTx > lastRx) ? lastTx : lastRx;
    double flowDuration = 0.0;

    if (flowEnd > firstTx) {
        flowDuration = flowEnd - firstTx;
    }

    double flowThroughput = (flowDuration > 0.0)
        ? (static_cast<double>(fs.rxBytes) * 8.0 / flowDuration)
        : 0.0;

    double flowAvgDelay = (fs.rxPackets > 0)
        ? (fs.delaySum.GetSeconds() / static_cast<double>(fs.rxPackets))
        : 0.0;

    double flowAvgJitter = (fs.rxPackets > 1)
        ? (fs.jitterSum.GetSeconds() / static_cast<double>(fs.rxPackets - 1))
        : 0.0;

    double flowLossRate = (fs.txPackets > 0)
        ? ((static_cast<double>(fs.txPackets - fs.rxPackets) / static_cast<double>(fs.txPackets)) * 100.0)
        : 0.0;

    flowDurations.push_back(flowDuration);
    flowThroughputs.push_back(flowThroughput);
    flowAvgDelays.push_back(flowAvgDelay);
    flowAvgJitters.push_back(flowAvgJitter);
    flowLossRates.push_back(flowLossRate);
}

		double pdr = (totalTxPackets > 0.0) ? (totalRxPackets / totalTxPackets) * 100.0 : 0.0;
		double plr = (totalTxPackets > 0.0) ? (totalLostPackets / totalTxPackets) * 100.0 : 0.0;
		double avgDelay = (totalRxPackets > 0.0) ? (totalDelay / totalRxPackets) : 0.0;
		double avgJitter = (totalRxPackets > 1.0) ? (totalJitter / (totalRxPackets - 1.0)) : 0.0;
		
		// average hop count per delivered packet
		double avgHopCount = (totalRxPackets > 0.0) ?
							((totalForwardings + totalRxPackets) / totalRxPackets) : 0.0;

		// throughput over the full measurement window, in bps
		double throughput = (measurementDuration > 0.0) ?
							((totalRxBytes * 8.0) / measurementDuration) : 0.0;

		double totalEnergyConsumed = 0.0;
		for (uint32_t i = 0; i < nodes.GetN(); ++i) {
			Ptr<energy::EnergySource> energySource = nodes.Get(i)->GetObject<energy::EnergySource>();  // ns-3.47: energy:: namespace
			if (energySource) {
				totalEnergyConsumed += (energySource->GetInitialEnergy() - energySource->GetRemainingEnergy());
			}
		}

		double energyEfficiency = (totalEnergyConsumed > 0.0) ?
								((totalRxBytes * 8.0) / totalEnergyConsumed) : 0.0;

		double routingMessages = static_cast<double>(g_helloCount + g_tcCount + g_midCount + g_hnaCount);

		double controlPacketRate = (measurementDuration > 0.0) ?
									(routingMessages / measurementDuration) : 0.0;
	
		double normalizedRoutingLoad = (totalTxPackets > 0.0) ?
									(routingMessages / totalTxPackets) : 0.0;

		double routingOverhead = ((routingMessages + totalTxPackets) > 0.0) ?
								(routingMessages / (routingMessages + totalTxPackets)) : 0.0;
		
		double routingOverheadBytesRatio =
				((g_olsrControlBytes + totalTxBytes) > 0.0) ?
				(static_cast<double>(g_olsrControlBytes) /
				static_cast<double>(g_olsrControlBytes + totalTxBytes)) : 0.0;
	 
		// Compute per-node MAC drop rate
		std::vector<double> macDropRates;
		macDropRates.reserve(macTxPerNode.size());

		for (uint32_t i = 0; i < macTxPerNode.size(); ++i) {
			uint32_t total = macTxPerNode[i] + macDropPerNode[i];
			double rate = (total > 0) ?
				static_cast<double>(macDropPerNode[i]) / static_cast<double>(total) : 0.0;
			macDropRates.push_back(rate);
		}

		// Avg
		double macDropRateAvg = ComputeMean(macDropRates);

		// Max
		double macDropRateMax = 0.0;
		for (std::vector<double>::const_iterator it = macDropRates.begin();
			 it != macDropRates.end(); ++it) {
			if (*it > macDropRateMax) macDropRateMax = *it;
		}

		// Variance (using existing ComputeStdDev)
		double macDropRateStd = ComputeStdDev(macDropRates, macDropRateAvg);

		double avgTcRows = (g_tcCount > 0) ?
						(static_cast<double>(g_totalTcRows) / static_cast<double>(g_tcCount)) : 0.0;

		double avgTxPacketsPerFlow = (flowCount > 0) ? 
						(totalTxPackets / static_cast<double>(flowCount)) : 0.0;

		double avgRxPacketsPerFlow = (flowCount > 0) ? 
						(totalRxPackets / static_cast<double>(flowCount)) : 0.0;

		double avgTxBytesPerFlow = (flowCount > 0) ? 
						(totalTxBytes / static_cast<double>(flowCount)) : 0.0;

		double avgRxBytesPerFlow = (flowCount > 0) ? 
						(totalRxBytes / static_cast<double>(flowCount)) : 0.0;

		double avgTxPacketSize = (totalTxPackets > 0.0) ? 
						(totalTxBytes / totalTxPackets) : 0.0;

		double avgRxPacketSize = (totalRxPackets > 0.0) ? 
						(totalRxBytes / totalRxPackets) : 0.0;

		double rxTxPacketRatio = (totalTxPackets > 0.0)	? 
						(totalRxPackets / totalTxPackets) : 0.0;

		double avgFlowDuration = ComputeMean(flowDurations);
		double flowDurationStd = ComputeStdDev(flowDurations, avgFlowDuration);

		double avgFlowThroughput = ComputeMean(flowThroughputs);
		double flowThroughputStd = ComputeStdDev(flowThroughputs, avgFlowThroughput);

		double avgFlowDelay = ComputeMean(flowAvgDelays);
		double flowDelayStd = ComputeStdDev(flowAvgDelays, avgFlowDelay);

		double avgFlowJitter = ComputeMean(flowAvgJitters);
		double flowJitterStd = ComputeStdDev(flowAvgJitters, avgFlowJitter);

		double avgFlowLossRate = ComputeMean(flowLossRates);
		double flowLossRateStd = ComputeStdDev(flowLossRates, avgFlowLossRate);

		std::vector<double> nodeSpeeds;
		for (uint32_t i = 0; i < nodes.GetN(); ++i) {
			Ptr<MobilityModel> mobilityModel = nodes.Get(i)->GetObject<MobilityModel>();
			double speed = 0.0;

			if (mobilityModel) {
				Vector velocity = mobilityModel->GetVelocity();
				speed = std::sqrt(velocity.x * velocity.x +
								  velocity.y * velocity.y +
								  velocity.z * velocity.z);
			}

			nodeSpeeds.push_back(speed);
		}

		double averageNodeSpeed = ComputeMean(nodeSpeeds);
		double nodeSpeedStd = ComputeStdDev(nodeSpeeds, averageNodeSpeed);

        // Write basic network performance metrics
        file << scenario << ",PacketDeliveryRatio," << ToString(pdr) << "," 
             << ToString(startTime) << "," << ToString(endTime) << "," << ToString(measurementDuration) << "\n";
        file << scenario << ",PacketLossRatio," << ToString(plr) << "," 
             << ToString(startTime) << "," << ToString(endTime) << "," << ToString(measurementDuration) << "\n";
        file << scenario << ",AverageEndToEndDelay," << ToString(avgDelay) << "," 
             << ToString(startTime) << "," << ToString(endTime) << "," << ToString(measurementDuration) << "\n";
        file << scenario << ",AverageJitter," << ToString(avgJitter) << "," 
             << ToString(startTime) << "," << ToString(endTime) << "," << ToString(measurementDuration) << "\n";
        file << scenario << ",Throughput," << ToString(throughput) << ","
             << ToString(startTime) << "," << ToString(endTime) << "," << ToString(measurementDuration) << "\n";

        // Per-window victim UDP-flow delivery (out of UDP_PACKETS_PER_WINDOW,
        // derived from the window length; 18 at the default 40 s window).
        // Unlike PacketDeliveryRatio (aggregated over ~51 flows incl. OLSR broadcast,
        // hence diluted), this isolates the single node1->node0 UDP stream -- the
        // metric that actually reveals a single-flow black-hole and whether the
        // defense restores delivery. g_udpReceivedAtWindowStart was snapshotted at
        // this window's start, so the delta is exactly this window's UDP receipts.
        uint32_t udpRxWindow = g_udpServer ? (g_udpServer->GetReceived() - g_udpReceivedAtWindowStart) : 0;
        double udpFlowPdr = (UDP_PACKETS_PER_WINDOW > 0) ? (100.0 * (double) udpRxWindow / (double) UDP_PACKETS_PER_WINDOW) : 0.0;
        file << scenario << ",UdpPacketsReceived," << ToString((double) udpRxWindow) << ","
             << ToString(startTime) << "," << ToString(endTime) << "," << ToString(measurementDuration) << "\n";
        file << scenario << ",UdpPacketsExpected," << ToString((double) UDP_PACKETS_PER_WINDOW) << ","
             << ToString(startTime) << "," << ToString(endTime) << "," << ToString(measurementDuration) << "\n";
        file << scenario << ",UdpFlowDeliveryRatio," << ToString(udpFlowPdr) << ","
             << ToString(startTime) << "," << ToString(endTime) << "," << ToString(measurementDuration) << "\n";
        file << scenario << ",AverageHopCount," << ToString(avgHopCount) << "," 
             << ToString(startTime) << "," << ToString(endTime) << "," << ToString(measurementDuration) << "\n";
        
        // Write additional network performance metrics
        file << scenario << ",TotalEnergyConsumption," << ToString(totalEnergyConsumed) << "," 
             << ToString(startTime) << "," << ToString(endTime) << "," << ToString(measurementDuration) << "\n";
        file << scenario << ",EnergyEfficiency," << ToString(energyEfficiency) << "," 
             << ToString(startTime) << "," << ToString(endTime) << "," << ToString(measurementDuration) << "\n";
        file << scenario << ",NormalizedRoutingLoad," << ToString(normalizedRoutingLoad) << "," 
             << ToString(startTime) << "," << ToString(endTime) << "," << ToString(measurementDuration) << "\n";
        file << scenario << ",RoutingOverheadRatio," << ToString(routingOverhead) << "," 
             << ToString(startTime) << "," << ToString(endTime) << "," << ToString(measurementDuration) << "\n";
		file << scenario << ",RoutingOverheadBytesRatio," << ToString(routingOverheadBytesRatio) << ","
			 << ToString(startTime) << "," << ToString(endTime) << "," << ToString(measurementDuration) << "\n";
		WriteScenarioMetric(file, scenario, "MACDropRateAvg",
                    macDropRateAvg, startTime, endTime, measurementDuration);
		WriteScenarioMetric(file, scenario, "MACDropRateMax",
                    macDropRateMax, startTime, endTime, measurementDuration);
		WriteScenarioMetric(file, scenario, "MACDropRateStd",
                    macDropRateStd, startTime, endTime, measurementDuration);
        file << scenario << ",AverageAdvertisedLinksPerTCMessage," << ToString(avgTcRows) << "," 
             << ToString(startTime) << "," << ToString(endTime) << "," << ToString(measurementDuration) << "\n";
        uint32_t totalMacTx = 0;
		uint32_t totalMacDrop = 0;
		for (uint32_t i = 0; i < macTxPerNode.size(); ++i) {
			totalMacTx  += macTxPerNode[i];
			totalMacDrop += macDropPerNode[i];
		}
		// Renamed 2026-08-03. Counts every frame handed to the MAC for
		// transmission, control and data alike -- not data packets only.
		// Previous labels for the identical quantity:
		//   "DataPacketRate"    - April 2026 dataset (simulations/features_*)
		//   "MacDataPacketRate" - source between April and 2026-08-03
		WriteScenarioMetric(file, scenario, "TransmissionRate",
							totalMacTx / measurementDuration,
							startTime, endTime, measurementDuration);
		WriteScenarioMetric(file, scenario, "MACDropPacketRate",
							totalMacDrop / measurementDuration,
							startTime, endTime, measurementDuration);
					
        // Add OLSR packet rates (convert total counts to per-second rates)
        file << scenario << ",HelloMessageRate," << ToString(g_helloCount / measurementDuration) << "," 
             << ToString(startTime) << "," << ToString(endTime) << "," << ToString(measurementDuration) << "\n";
        file << scenario << ",TcMessageRate," << ToString(g_tcCount / measurementDuration) << "," 
             << ToString(startTime) << "," << ToString(endTime) << "," << ToString(measurementDuration) << "\n";
        file << scenario << ",MidMessageRate," << ToString(g_midCount / measurementDuration) << "," 
             << ToString(startTime) << "," << ToString(endTime) << "," << ToString(measurementDuration) << "\n";
        file << scenario << ",HnaMessageRate," << ToString(g_hnaCount / measurementDuration) << "," 
             << ToString(startTime) << "," << ToString(endTime) << "," << ToString(measurementDuration) << "\n";
        file << scenario << ",ControlPacketRate," << ToString(controlPacketRate) << ","
			 << ToString(startTime) << "," << ToString(endTime) << "," << ToString(measurementDuration) << "\n";
        // Add MAC packet rates
		WriteScenarioMetric(file, scenario, "FlowCount", static_cast<double>(flowCount),
                    startTime, endTime, measurementDuration);
		WriteScenarioMetric(file, scenario, "AvgFlowDuration", avgFlowDuration,
							startTime, endTime, measurementDuration);
		WriteScenarioMetric(file, scenario, "FlowDurationStd", flowDurationStd,
							startTime, endTime, measurementDuration);
		WriteScenarioMetric(file, scenario, "AvgFlowThroughput", avgFlowThroughput,
							startTime, endTime, measurementDuration);
		WriteScenarioMetric(file, scenario, "FlowThroughputStd", flowThroughputStd,
							startTime, endTime, measurementDuration);
		WriteScenarioMetric(file, scenario, "AvgFlowDelay", avgFlowDelay,
							startTime, endTime, measurementDuration);
		WriteScenarioMetric(file, scenario, "FlowDelayStd", flowDelayStd,
							startTime, endTime, measurementDuration);
		WriteScenarioMetric(file, scenario, "AvgFlowJitter", avgFlowJitter,
							startTime, endTime, measurementDuration);
		WriteScenarioMetric(file, scenario, "FlowJitterStd", flowJitterStd,
							startTime, endTime, measurementDuration);
		WriteScenarioMetric(file, scenario, "AvgFlowLossRate", avgFlowLossRate,
							startTime, endTime, measurementDuration);
		WriteScenarioMetric(file, scenario, "FlowLossRateStd", flowLossRateStd,
							startTime, endTime, measurementDuration);
		WriteScenarioMetric(file, scenario, "AvgTxPacketsPerFlow", avgTxPacketsPerFlow,
							startTime, endTime, measurementDuration);
		WriteScenarioMetric(file, scenario, "AvgRxPacketsPerFlow", avgRxPacketsPerFlow,
							startTime, endTime, measurementDuration);
		WriteScenarioMetric(file, scenario, "AvgTxBytesPerFlow", avgTxBytesPerFlow,
							startTime, endTime, measurementDuration);
		WriteScenarioMetric(file, scenario, "AvgRxBytesPerFlow", avgRxBytesPerFlow,
							startTime, endTime, measurementDuration);
		WriteScenarioMetric(file, scenario, "AvgTxPacketSize", avgTxPacketSize,
							startTime, endTime, measurementDuration);
		WriteScenarioMetric(file, scenario, "AvgRxPacketSize", avgRxPacketSize,
							startTime, endTime, measurementDuration);
		WriteScenarioMetric(file, scenario, "RxTxPacketRatio", rxTxPacketRatio,
							startTime, endTime, measurementDuration);
		WriteScenarioMetric(file, scenario, "AverageNodeSpeed", averageNodeSpeed,
							startTime, endTime, measurementDuration);
		WriteScenarioMetric(file, scenario, "NodeSpeedStd", nodeSpeedStd,
							startTime, endTime, measurementDuration);
		// Per-node OLSR state metrics
		double totalNeighbors         = 0.0;
		double totalMprs              = 0.0;
		double totalRoutingTableSize  = 0.0;
		//uint32_t fictiveRequiredCount = 0;
		//uint32_t fictiveDeclaredCount = 0;
		uint32_t validNodes           = 0;

		for (uint32_t i = 0; i < nodes.GetN(); ++i) {
			Ptr<RoutingProtocol> rp = nodes.Get(i)->GetObject<RoutingProtocol>();
			if (!rp) continue;
			++validNodes;
			totalNeighbors        += static_cast<double>(rp->getNeighborsSize());
			totalMprs             += static_cast<double>(rp->getMprSize());
			totalRoutingTableSize += static_cast<double>(rp->getRoutingTableSize());
			//if (rp->tcHelloInvalidated_FictiveRequired) ++fictiveRequiredCount;
			//if (rp->returnDeclaredFictive())            ++fictiveDeclaredCount;
		}

		double avgNeighborCount     = (validNodes > 0) ? totalNeighbors        / validNodes : 0.0;
		double avgMprCount          = (validNodes > 0) ? totalMprs             / validNodes : 0.0;
		double avgRoutingTableSize  = (validNodes > 0) ? totalRoutingTableSize / validNodes : 0.0;
		double mprNeighborRatio     = (totalNeighbors > 0.0) ? totalMprs / totalNeighbors  : 0.0;
		/*
		double fictiveRequiredRatio = (validNodes > 0) ?
			static_cast<double>(fictiveRequiredCount) / validNodes : 0.0;
		double fictiveDeclaredRatio = (validNodes > 0) ?
			static_cast<double>(fictiveDeclaredCount) / validNodes : 0.0;
		*/
		WriteScenarioMetric(file, scenario, "AverageNeighborCount",    avgNeighborCount,    startTime, endTime, measurementDuration);
		WriteScenarioMetric(file, scenario, "AverageMprCount",         avgMprCount,         startTime, endTime, measurementDuration);
		WriteScenarioMetric(file, scenario, "AverageRoutingTableSize", avgRoutingTableSize, startTime, endTime, measurementDuration);
		WriteScenarioMetric(file, scenario, "MprNeighborRatio",        mprNeighborRatio,    startTime, endTime, measurementDuration);
		/*
		WriteScenarioMetric(file, scenario, "FictiveRequiredRatio",    fictiveRequiredRatio,startTime, endTime, measurementDuration);
		WriteScenarioMetric(file, scenario, "FictiveDeclaredRatio",    fictiveDeclaredRatio,startTime, endTime, measurementDuration);
		*/

		// ==================================================================
		// CORRECTED GLOBAL METRICS (v4). Every original row above is written
		// unchanged; each corrected recomputation is a NEW row prefixed "X_".
		// All X_ values are derivable from captured frames alone.
		// ==================================================================
		{
			// ---- the single UDP data flow (node1 -> node0, dst port 80) ----
			double dTxPk = 0.0, dRxPk = 0.0, dTxB = 0.0, dRxB = 0.0;
			double dDelaySum = 0.0, dJitterSum = 0.0, dFwd = 0.0;
			for (it = stats.begin(); it != stats.end(); ++it) {
				Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(it->first);
				if (t.protocol == 17 && t.destinationPort == 80) {
					dTxPk += it->second.txPackets;
					dRxPk += it->second.rxPackets;
					dTxB  += it->second.txBytes;
					dRxB  += it->second.rxBytes;
					dDelaySum  += it->second.delaySum.GetSeconds();
					dJitterSum += it->second.jitterSum.GetSeconds();
					dFwd  += it->second.timesForwarded;
				}
			}

			// Delivery of the data flow against the application schedule. The
			// denominator is the scheduled count, so a source-side black hole
			// reads 0 instead of vanishing with the isActiveFlow filter.
			double xPdr = (UDP_PACKETS_PER_WINDOW > 0)
				? (100.0 * (double) udpRxWindow / (double) UDP_PACKETS_PER_WINDOW) : 0.0;
			WriteScenarioMetric(file, scenario, "X_PacketDeliveryRatio", xPdr,
								startTime, endTime, measurementDuration);
			WriteScenarioMetric(file, scenario, "X_PacketLossRatio", 100.0 - xPdr,
								startTime, endTime, measurementDuration);

			// Numerator and denominator from the same population: the data flow.
			double xHops = (dRxPk > 0.0) ? (dFwd / dRxPk) + 1.0 : 0.0;
			WriteScenarioMetric(file, scenario, "X_AverageHopCount", xHops,
								startTime, endTime, measurementDuration);

			double xDelay  = (dRxPk > 0.0) ? (dDelaySum / dRxPk) : 0.0;
			double xJitter = (dRxPk > 1.0) ? (dJitterSum / (dRxPk - 1.0)) : 0.0;
			WriteScenarioMetric(file, scenario, "X_AverageEndToEndDelay", xDelay,
								startTime, endTime, measurementDuration);
			WriteScenarioMetric(file, scenario, "X_AverageJitter", xJitter,
								startTime, endTime, measurementDuration);

			// Goodput, not control chatter.
			double xDataThroughput = (measurementDuration > 0.0)
				? (dRxB * 8.0 / measurementDuration) : 0.0;
			WriteScenarioMetric(file, scenario, "X_DataThroughput", xDataThroughput,
								startTime, endTime, measurementDuration);

			// Overhead ratios against data-plane quantities only. The scheduled
			// UDP count keeps the denominator defined under a full block.
			double xNrl = routingMessages / (double) UDP_PACKETS_PER_WINDOW;
			WriteScenarioMetric(file, scenario, "X_NormalizedRoutingLoad", xNrl,
								startTime, endTime, measurementDuration);
			double xRor = ((routingMessages + (double) UDP_PACKETS_PER_WINDOW) > 0.0)
				? (routingMessages / (routingMessages + (double) UDP_PACKETS_PER_WINDOW)) : 0.0;
			WriteScenarioMetric(file, scenario, "X_RoutingOverheadRatio", xRor,
								startTime, endTime, measurementDuration);
			// Control bytes over control-plus-data bytes. The original folded the
			// control bytes into both numerator and denominator, saturating at 0.5.
			double xRobr = ((g_olsrControlBytes + dTxB) > 0.0)
				? ((double) g_olsrControlBytes / ((double) g_olsrControlBytes + dTxB)) : 0.0;
			WriteScenarioMetric(file, scenario, "X_RoutingOverheadBytesRatio", xRobr,
								startTime, endTime, measurementDuration);

			// Generation counts, not flood-transmission counts.
			double xTcGen = (measurementDuration > 0.0)
				? ((double) g_tcUniqueCount / measurementDuration) : 0.0;
			WriteScenarioMetric(file, scenario, "X_TcGenerationRate", xTcGen,
								startTime, endTime, measurementDuration);
			double xAdvUnique = (g_tcUniqueCount > 0)
				? ((double) g_tcUniqueRows / (double) g_tcUniqueCount) : 0.0;
			WriteScenarioMetric(file, scenario, "X_AvgAdvertisedLinksPerUniqueTc", xAdvUnique,
								startTime, endTime, measurementDuration);

			// The air truth: frames whose PHY transmission began, everything included.
			double xTxRate = (measurementDuration > 0.0)
				? ((double) g_phyTxFrames / measurementDuration) : 0.0;
			WriteScenarioMetric(file, scenario, "X_TransmissionRate", xTxRate,
								startTime, endTime, measurementDuration);
			double xTxBytesRate = (measurementDuration > 0.0)
				? ((double) g_phyTxBytes * 8.0 / measurementDuration) : 0.0;
			WriteScenarioMetric(file, scenario, "X_TransmissionByteRate", xTxBytesRate,
								startTime, endTime, measurementDuration);

			// MPR average as the air shows it: sum of the last advertised TC set
			// size per originator, divided by the node count AS THE AIR SHOWS IT
			// (g_addrsSeenGlobal = every address seen originating or advertised in
			// a TC, so the fictitious nodes the defence injects enter BOTH the
			// numerator and the denominator). Dividing by 50 instead would mix a
			// numerator that counts the fictitious links with a denominator that
			// does not -- the population mismatch §3.2 documents. This single
			// perceived-denominator form is the only one emitted.
			double advSum = 0.0;
			for (std::map<uint32_t,uint32_t>::const_iterator ai = g_lastAdvByOriginator.begin();
				 ai != g_lastAdvByOriginator.end(); ++ai) {
				advSum += (double) ai->second;
			}
			double xAddrsSeen = (double) g_addrsSeenGlobal.size();
			double xMprPerceived = (xAddrsSeen > 0.0) ? (advSum / xAddrsSeen) : 0.0;
			WriteScenarioMetric(file, scenario, "X_AverageMprCountPerceived", xMprPerceived,
								startTime, endTime, measurementDuration);
		}
    }
    
    file.close();
    NS_LOG_INFO("Scenario metrics saved to: " << filename);
}

// ---------------------------------------------------------------------------
// Per-vantage-point metrics. Written to a separate file so the global CSV keeps
// its exact previous format. One row per observer per window; the K = 1,2,3,5,10
// subsets are formed offline by pooling rows.
// ---------------------------------------------------------------------------
static std::string AddrStr(uint32_t a) {
    std::ostringstream o;
    o << ((a>>24)&0xFF) << "." << ((a>>16)&0xFF) << "." << ((a>>8)&0xFF) << "." << (a&0xFF);
    return o.str();
}

// ---------------------------------------------------------------------------
// Per-vantage-point metrics. Two files are written next to the unchanged global
// CSV, which keeps its previous format exactly.
//
//   observer_metrics-N.csv : one summary row per observer per window.
//   observer_detail-N.csv  : the RAW per-observer maps, packed one row each.
//
// The detail file exists because the K-observer collusion estimates are set
// operations, not sums of the summary scalars. Pooling rules to apply offline:
//   ADDRS : set union over the K observers.
//   ADV   : per originator, take the record with the largest LastTime, since
//           that observer heard the most recent advertisement.
//   SRC   : per source, take the MAX rate over the K observers. A source
//           transmits at one rate; each in-range observer sees at most that,
//           so the best receiver gives the best estimate. Summing would double
//           count frames overheard by several observers.
// ---------------------------------------------------------------------------
static void SaveObserverMetrics(const std::string& scenario,
                                double startTime, double endTime) {
    if (g_obs.empty()) return;

    double duration = endTime - startTime;

    // ---------------- summary ----------------
    std::string filename = g_outputBaseDir + scenario + "/observer_metrics-" +
                           ToString(g_currentRun) + ".csv";
    std::ofstream f(filename.c_str());
    if (!f.is_open()) {
        NS_LOG_ERROR("Cannot open observer file: " << filename);
        return;
    }

    f << "Scenario,ObserverIdx,ObserverNodeId,RunAttackerNodeId,ObserverIsAttacker,"
      << "SniffedFrames,SniffedDataFrames,SniffedBytes,SniffedFrameRate,SniffedDataFrameRate,"
      << "TcCount,TcRows,AvgAdvertisedLinksPerTC,OriginatorsHeard,SumAdvertisedLinks,"
      << "AvgAdvLinksPerOriginator,DistinctAddrsSeen,OutOfRangeAddrsSeen,"
      << "HelloCount,MidCount,HnaCount,SourcesHeard,SrcRateStd,"
      << "StartTime,EndTime,Duration,"
      // v4 listener extensions. Appended after the original columns so existing
      // name-based parsers keep working.
      << "ObsDataBytes,ObsOlsrFrames,ObsOlsrBytes,ObsControlBytesRatio,"
      << "ObsNonOlsrDataFrames,ObsAvgTcHopCount,ObsTcUniqueCount,ObsTcGenerationRate,"
      << "ObsAvgAdvLinksPerUniqueTc,ObsSrcDurationMean,ObsSrcDurationStd,"
      << "ObsSrcIatMean,ObsSrcIatStd,ObsHopDelayCount,ObsHopDelayMean,ObsHopDelayStd,"
      << "ObsAvgFramesPerSource,ObsAvgBytesPerSource\n";
    f << std::fixed << std::setprecision(6);

    // K vantage observers, then ONE extra iteration for the network-wide observer
    // g_globalObs (emitted with ObserverIdx = -1). Identical code and formulas;
    // the global row differs only in coverage.
    for (size_t oi = 0; oi <= g_obs.size(); ++oi) {
        const bool isGlobal = (oi == g_obs.size());
        const ObserverCounters& oc = isGlobal ? g_globalObs : g_obs[oi];
        const long obsIdxOut  = isGlobal ? -1L : (long) oi;
        const long obsNodeOut = isGlobal ? -1L : (long) g_observerNodes[oi];

        double frameRate     = (duration > 0.0) ? (double) oc.frames     / duration : 0.0;
        double dataFrameRate = (duration > 0.0) ? (double) oc.dataFrames / duration : 0.0;

        double avgAdvPerTc = (oc.tcCount > 0)
            ? (double) oc.tcRows / (double) oc.tcCount : 0.0;

        double sumAdv = 0.0;
        for (std::map<uint32_t,uint32_t>::const_iterator it = oc.advByOriginator.begin();
             it != oc.advByOriginator.end(); ++it) {
            sumAdv += (double) it->second;
        }
        uint32_t nOrig = (uint32_t) oc.advByOriginator.size();

        // Mean advertised links per originator HEARD. This is NOT a reconstruction
        // of the global AverageMprCount: under an active defence the advertised set
        // carries the injected fictitious link, which the node's real MPR-selector
        // set does not. Reported as its own observable quantity.
        double avgAdvPerOrig = (nOrig > 0) ? sumAdv / (double) nOrig : 0.0;

        // DIAGNOSTIC ONLY. Addresses outside the deployment's own /24 are the
        // injected fictitious nodes (getFakeAddress adds 65536). A real defence
        // would not brand its decoys with a distinguishable prefix, so this count
        // must not be fed to the classifier; it is here to measure how much of the
        // observable signal is an artifact of that addressing choice.
        uint32_t outOfRange = 0;
        for (std::set<uint32_t>::const_iterator it = oc.addrsSeen.begin();
             it != oc.addrsSeen.end(); ++it) {
            if (((*it) & 0xFFFFFF00u) != 0x0A000000u) ++outOfRange;
        }

        // Dispersion of per-source byte rates over the sources this observer heard.
        // Related to, but NOT the same statistic as, the global FlowThroughputStd:
        // that one is computed by FlowMonitor over 51 classified flows using rxBytes
        // accumulated at the receivers, whereas this is computed over the sources a
        // single vantage point overheard, using full frame sizes.
        std::vector<double> rates;
        for (std::map<uint32_t,uint64_t>::const_iterator it = oc.bytesBySource.begin();
             it != oc.bytesBySource.end(); ++it) {
            rates.push_back((duration > 0.0)
                            ? ((double) it->second * 8.0 / duration) : 0.0);
        }
        double mean = 0.0, srcStd = 0.0;
        if (!rates.empty()) {
            for (size_t i = 0; i < rates.size(); ++i) mean += rates[i];
            mean /= (double) rates.size();
            for (size_t i = 0; i < rates.size(); ++i) {
                double d = rates[i] - mean;
                srcStd += d * d;
            }
            srcStd = std::sqrt(srcStd / (double) rates.size());
        }

        int isAttacker = (!isGlobal && g_attackerNodeId >= 0 &&
                          g_observerNodes[oi] == (uint32_t) g_attackerNodeId) ? 1 : 0;

        // ---- v4 listener extension values ----
        double obsCtrlBytesRatio = (oc.dataBytes > 0)
            ? ((double) oc.olsrBytes / (double) oc.dataBytes) : 0.0;
        uint64_t nonOlsrData = (oc.dataFrames >= oc.olsrFrames)
            ? (oc.dataFrames - oc.olsrFrames) : 0;
        double avgTcHop = (oc.tcCount > 0)
            ? ((double) oc.tcHopSum / (double) oc.tcCount) : 0.0;
        double tcGenRate = (duration > 0.0)
            ? ((double) oc.tcUniqueCount / duration) : 0.0;
        double advPerUnique = (oc.tcUniqueCount > 0)
            ? ((double) oc.tcUniqueRows / (double) oc.tcUniqueCount) : 0.0;

        // Per-source visibility durations (listener analogue of flow duration).
        std::vector<double> durs;
        for (std::map<uint32_t,double>::const_iterator fi = oc.firstSeenBySource.begin();
             fi != oc.firstSeenBySource.end(); ++fi) {
            std::map<uint32_t,double>::const_iterator la = oc.lastSeenBySource.find(fi->first);
            if (la != oc.lastSeenBySource.end())
                durs.push_back(la->second - fi->second);
        }
        double durMean = ComputeMean(durs);
        double durStd  = ComputeStdDev(durs, durMean);

        // Inter-arrival statistics: mean over sources of the per-source mean,
        // and mean over sources of the per-source std (the jitter analogue).
        std::vector<double> iatMeans, iatStds;
        for (std::map<uint32_t,WelfordAcc>::const_iterator wi = oc.iatBySource.begin();
             wi != oc.iatBySource.end(); ++wi) {
            if (wi->second.n > 0) {
                iatMeans.push_back(wi->second.mean);
                iatStds.push_back(wi->second.Std());
            }
        }
        double iatMean = ComputeMean(iatMeans);
        double iatStd  = ComputeMean(iatStds);

        double avgFramesPerSrc = (!oc.framesBySource.empty())
            ? ((double) oc.dataFrames / (double) oc.framesBySource.size()) : 0.0;
        double avgBytesPerSrc = (!oc.bytesBySource.empty())
            ? ((double) oc.dataBytes / (double) oc.bytesBySource.size()) : 0.0;

        f << scenario << ","
          << obsIdxOut << ","
          << obsNodeOut << ","
          << g_attackerNodeId << ","
          << isAttacker << ","
          << oc.frames << ","
          << oc.dataFrames << ","
          << oc.bytes << ","
          << ToString(frameRate) << ","
          << ToString(dataFrameRate) << ","
          << oc.tcCount << ","
          << oc.tcRows << ","
          << ToString(avgAdvPerTc) << ","
          << nOrig << ","
          << ToString(sumAdv) << ","
          << ToString(avgAdvPerOrig) << ","
          << oc.addrsSeen.size() << ","
          << outOfRange << ","
          << oc.helloCount << ","
          << oc.midCount << ","
          << oc.hnaCount << ","
          << rates.size() << ","
          << ToString(srcStd) << ","
          << ToString(startTime) << ","
          << ToString(endTime) << ","
          << ToString(duration) << ","
          << oc.dataBytes << ","
          << oc.olsrFrames << ","
          << oc.olsrBytes << ","
          << ToString(obsCtrlBytesRatio) << ","
          << nonOlsrData << ","
          << ToString(avgTcHop) << ","
          << oc.tcUniqueCount << ","
          << ToString(tcGenRate) << ","
          << ToString(advPerUnique) << ","
          << ToString(durMean) << ","
          << ToString(durStd) << ","
          << ToString(iatMean) << ","
          << ToString(iatStd) << ","
          << oc.hopDelay.n << ","
          << ToString(oc.hopDelay.mean) << ","
          << ToString(oc.hopDelay.Std()) << ","
          << ToString(avgFramesPerSrc) << ","
          << ToString(avgBytesPerSrc) << "\n";
    }
    f.close();

    // ---------------- raw maps, for offline K-pooling ----------------
    std::string dfilename = g_outputBaseDir + scenario + "/observer_detail-" +
                            ToString(g_currentRun) + ".csv";
    std::ofstream d(dfilename.c_str());
    if (!d.is_open()) {
        NS_LOG_ERROR("Cannot open observer detail file: " << dfilename);
        return;
    }
    d << "Scenario,ObserverIdx,ObserverNodeId,RecordType,Count,Payload\n";
    d << std::fixed << std::setprecision(6);

    for (size_t oi = 0; oi < g_obs.size(); ++oi) {
        const ObserverCounters& oc = g_obs[oi];
        std::ostringstream a, v, r;

        // ADDRS: addr;addr;...
        for (std::set<uint32_t>::const_iterator it = oc.addrsSeen.begin();
             it != oc.addrsSeen.end(); ++it) {
            if (it != oc.addrsSeen.begin()) a << ";";
            a << AddrStr(*it);
        }
        // ADV: originator:advCount:lastTimeSec;...
        for (std::map<uint32_t,uint32_t>::const_iterator it = oc.advByOriginator.begin();
             it != oc.advByOriginator.end(); ++it) {
            if (it != oc.advByOriginator.begin()) v << ";";
            std::map<uint32_t,double>::const_iterator t = oc.advTimeByOriginator.find(it->first);
            double lastT = (t != oc.advTimeByOriginator.end()) ? t->second : -1.0;
            v << AddrStr(it->first) << ":" << it->second << ":" << lastT;
        }
        // SRC: source:bytes:frames;...
        for (std::map<uint32_t,uint64_t>::const_iterator it = oc.bytesBySource.begin();
             it != oc.bytesBySource.end(); ++it) {
            if (it != oc.bytesBySource.begin()) r << ";";
            std::map<uint32_t,uint64_t>::const_iterator fr = oc.framesBySource.find(it->first);
            uint64_t nf = (fr != oc.framesBySource.end()) ? fr->second : 0;
            r << AddrStr(it->first) << ":" << it->second << ":" << nf;
        }

        d << scenario << "," << oi << "," << g_observerNodes[oi] << ",ADDRS,"
          << oc.addrsSeen.size()        << "," << a.str() << "\n";
        d << scenario << "," << oi << "," << g_observerNodes[oi] << ",ADV,"
          << oc.advByOriginator.size()  << "," << v.str() << "\n";
        d << scenario << "," << oi << "," << g_observerNodes[oi] << ",SRC,"
          << oc.bytesBySource.size()    << "," << r.str() << "\n";

        // v4 TIM: source:firstSeen:lastSeen:frames;...  K-observer pooling rule:
        // per source take min(firstSeen) and max(lastSeen) over the K observers.
        std::ostringstream tm;
        for (std::map<uint32_t,double>::const_iterator fi = oc.firstSeenBySource.begin();
             fi != oc.firstSeenBySource.end(); ++fi) {
            if (fi != oc.firstSeenBySource.begin()) tm << ";";
            std::map<uint32_t,double>::const_iterator la = oc.lastSeenBySource.find(fi->first);
            std::map<uint32_t,uint64_t>::const_iterator fr = oc.framesBySource.find(fi->first);
            double lastT = (la != oc.lastSeenBySource.end()) ? la->second : fi->second;
            uint64_t nf  = (fr != oc.framesBySource.end()) ? fr->second : 0;
            tm << AddrStr(fi->first) << ":" << ToString(fi->second) << ":" << ToString(lastT) << ":" << nf;
        }
        d << scenario << "," << oi << "," << g_observerNodes[oi] << ",TIM,"
          << oc.firstSeenBySource.size() << "," << tm.str() << "\n";
    }
    d.close();
    NS_LOG_INFO("Observer metrics saved to: " << filename);
}

// Function to reset OLSR packet counters
static void ResetOlsrCounters() {
    g_helloCount = 0;
    g_tcCount = 0;
    g_midCount = 0;
    g_totalTcRows = 0;
    g_olsrControlBytes = 0;
    g_hnaCount = 0;
    std::fill(macTxPerNode.begin(), macTxPerNode.end(), 0);
    std::fill(macDropPerNode.begin(), macDropPerNode.end(), 0);
    // v4 corrected-global counters
    g_phyTxFrames = 0;
    g_phyTxBytes  = 0;
    g_lastAdvByOriginator.clear();
    g_tcSeenGlobal.clear();
    g_tcUniqueCount = 0;
    g_tcUniqueRows  = 0;
    g_addrsSeenGlobal.clear();
    for (size_t oi = 0; oi < g_obs.size(); ++oi) g_obs[oi].Reset();
    g_globalObs.Reset();
    NS_LOG_INFO("OLSR counters reset at time " << Simulator::Now().GetSeconds());
}

static void EndBaselineMeasurement(NodeContainer* nodes) {
    NS_LOG_INFO("Ending baseline measurement at " << Simulator::Now().GetSeconds() << "s");
    SaveScenarioMetrics("baseline", *nodes, BASELINE_START, BASELINE_END);
    SaveObserverMetrics("baseline", BASELINE_START, BASELINE_END);
}

static void StartBaselineMeasurement(NodeContainer* nodes, Ptr<UdpServer> udpServer) {
    NS_LOG_INFO("Starting baseline measurement at " << Simulator::Now().GetSeconds() << "s");
    SnapshotUdpReceived(udpServer);
    ResetOlsrCounters();
    if (g_flowMonitor) {
        g_flowMonitor->ResetAllStats();
    }
    Simulator::Schedule(Seconds(MEASUREMENT_DURATION), &EndBaselineMeasurement, nodes);
}

static void EndAttackOnlyMeasurement(NodeContainer* nodes) {
    NS_LOG_INFO("Ending attack_only measurement at " << Simulator::Now().GetSeconds() << "s");
    SaveScenarioMetrics("attack_only", *nodes, ATTACK_ONLY_START, ATTACK_ONLY_END);
    SaveObserverMetrics("attack_only", ATTACK_ONLY_START, ATTACK_ONLY_END);
}

static void StartAttackOnlyMeasurement(NodeContainer* nodes, Ptr<UdpServer> udpServer) {
    NS_LOG_INFO("Starting attack_only measurement at " << Simulator::Now().GetSeconds() << "s");
    SnapshotUdpReceived(udpServer);
	ResetOlsrCounters();
    if (g_flowMonitor) {
        g_flowMonitor->ResetAllStats();
    }
    Simulator::Schedule(Seconds(MEASUREMENT_DURATION), &EndAttackOnlyMeasurement, nodes);
}

static void EndDefenseOnlyMeasurement(NodeContainer* nodes) {
    NS_LOG_INFO("Ending defense_only measurement at " << Simulator::Now().GetSeconds() << "s");
    SaveScenarioMetrics("defense_only", *nodes, DEFENSE_ONLY_START, DEFENSE_ONLY_END);
    SaveObserverMetrics("defense_only", DEFENSE_ONLY_START, DEFENSE_ONLY_END);
}

static void StartDefenseOnlyMeasurement(NodeContainer* nodes, Ptr<UdpServer> udpServer) {
    NS_LOG_INFO("Starting defense_only measurement at " << Simulator::Now().GetSeconds() << "s");
    SnapshotUdpReceived(udpServer);    
	ResetOlsrCounters();
    if (g_flowMonitor) {
        g_flowMonitor->ResetAllStats();
    }
    Simulator::Schedule(Seconds(MEASUREMENT_DURATION), &EndDefenseOnlyMeasurement, nodes);
}

static void EndDefenseAttackMeasurement(NodeContainer* nodes) {
    NS_LOG_INFO("Ending defense_vs_attack measurement at " << Simulator::Now().GetSeconds() << "s");
    SaveScenarioMetrics("defense_vs_attack", *nodes, DEFENSE_ATTACK_START, DEFENSE_ATTACK_END);
    SaveObserverMetrics("defense_vs_attack", DEFENSE_ATTACK_START, DEFENSE_ATTACK_END);
}

static void StartDefenseAttackMeasurement(NodeContainer* nodes, Ptr<UdpServer> udpServer) {
    NS_LOG_INFO("Starting defense_vs_attack measurement at " << Simulator::Now().GetSeconds() << "s");
    SnapshotUdpReceived(udpServer);
    ResetOlsrCounters();
    if (g_flowMonitor) {
        g_flowMonitor->ResetAllStats();
    }
    Simulator::Schedule(Seconds(MEASUREMENT_DURATION), &EndDefenseAttackMeasurement, nodes);
}

// ---------------------------------------------------------------------------
// Window-order experiment (--scenarioOrder, STATE §23.10 / §25.16).
// Parameterised twins of the four Start*/End*Measurement pairs above: same
// snapshot/reset/flow-reset sequence, but the phase label and window times
// arrive as arguments, so any phase can run in any slot. Used ONLY when
// --scenarioOrder is set; the fixed-order path never calls these.
struct PhaseSpec { const char* label; bool attackOn; bool defenseOn; };
static const PhaseSpec PHASES[4] = {
    {"baseline",          false, false},
    {"attack_only",       true,  false},
    {"defense_only",      false, true },
    {"defense_vs_attack", true,  true },
};

static void EndPhaseMeasurement(NodeContainer* nodes, uint32_t phaseIdx,
                                double startT, double endT) {
    NS_LOG_INFO("Ending " << PHASES[phaseIdx].label << " measurement at "
                << Simulator::Now().GetSeconds() << "s");
    SaveScenarioMetrics(PHASES[phaseIdx].label, *nodes, startT, endT);
    SaveObserverMetrics(PHASES[phaseIdx].label, startT, endT);
}

static void StartPhaseMeasurement(NodeContainer* nodes, Ptr<UdpServer> udpServer,
                                  uint32_t phaseIdx, double startT, double endT) {
    NS_LOG_INFO("Starting " << PHASES[phaseIdx].label << " measurement at "
                << Simulator::Now().GetSeconds() << "s");
    SnapshotUdpReceived(udpServer);
    ResetOlsrCounters();
    if (g_flowMonitor) {
        g_flowMonitor->ResetAllStats();
    }
    Simulator::Schedule(Seconds(MEASUREMENT_DURATION), &EndPhaseMeasurement,
                        nodes, phaseIdx, startT, endT);
}

// Enhanced connectivity check function
static void CheckAndReportConnectivity(NodeContainer* cont) {
    uint32_t fullyConnectedNodes = 0;
    for (uint32_t i = 0; i < cont->GetN(); ++i) {
        uint32_t routingTableSize = cont->Get(i)->GetObject<RoutingProtocol>()->getRoutingTableSize();
        if (routingTableSize == cont->GetN() - 1) {
            fullyConnectedNodes++;
        }
    }
    
    double connectivityRatio = (double)fullyConnectedNodes / cont->GetN();
    NS_LOG_INFO("Connectivity check: " << fullyConnectedNodes << "/" << cont->GetN() 
                << " nodes fully connected (" << (connectivityRatio * 100) << "%)");
    
    if (connectivityRatio < 0.8) { // 80% threshold
        NS_LOG_WARN("Poor connectivity detected! Ratio: " << connectivityRatio);
    }
}

// ------------------------------------------------------------
// Topology probe for IV-B bias analysis
// ------------------------------------------------------------
// Runs at t=59s (before AssertConnectivity at t=60s) for every seed, so
// accepted AND rejected runs both produce a row. Output file is shared
// across parallel workers; writes are serialized with flock(LOCK_EX).
// The probe only reads simulator state (routing tables, neighbor sets,
// mobility positions) and appends to a file, so it does not perturb
// ns-3 event ordering or determinism.

static void WriteTopologyProbeRow(const std::string& row) {
    if (g_topologyProbeFile.empty()) return;
    int fd = open(g_topologyProbeFile.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (fd < 0) {
        std::cerr << "[topology_probe] open failed for " << g_topologyProbeFile
                  << " (errno=" << errno << ")" << std::endl;
        return;
    }
    if (flock(fd, LOCK_EX) < 0) {
        std::cerr << "[topology_probe] flock failed (errno=" << errno << ")" << std::endl;
        close(fd);
        return;
    }
    // Write header if file is empty. Safe under LOCK_EX.
    off_t sz = lseek(fd, 0, SEEK_END);
    if (sz == 0) {
        const char* hdr =
            "seed,run_id,mobility,"
            "avg_neighbor_count,std_neighbor_count,min_neighbor_count,max_neighbor_count,"
            "avg_two_hop_count,std_two_hop_count,"
            "avg_routing_table_size,std_routing_table_size,"
            "fully_converged_nodes,node1_is_neighbor_of_node0,"
            "avg_min_euclidean_dist\n";
        ssize_t hw = write(fd, hdr, std::strlen(hdr));
        (void)hw;
    }
    ssize_t w = write(fd, row.c_str(), row.size());
    (void)w;
    flock(fd, LOCK_UN);
    close(fd);
}

static void RecordTopologyProbe(NodeContainer* cont) {
    if (g_topologyProbeFile.empty()) return;
    if (cont == nullptr || cont->GetN() == 0) return;

    const uint32_t N = cont->GetN();

    std::vector<double> neighborCounts;
    std::vector<double> twoHopCounts;
    std::vector<double> rtSizes;
    uint32_t fullyConverged = 0;

    double minNei = std::numeric_limits<double>::infinity();
    double maxNei = -std::numeric_limits<double>::infinity();

    for (uint32_t i = 0; i < N; ++i) {
        Ptr<RoutingProtocol> rp = cont->Get(i)->GetObject<RoutingProtocol>();
        if (!rp) continue;
        const NeighborSet& nei = rp->getNeighborSet();
        const TwoHopNeighborSet& two = rp->getTwoHopNeighborSet();
        uint32_t rtSize = rp->getRoutingTableSize();

        double nc = static_cast<double>(nei.size());
        double tc = static_cast<double>(two.size());
        double rs = static_cast<double>(rtSize);

        neighborCounts.push_back(nc);
        twoHopCounts.push_back(tc);
        rtSizes.push_back(rs);

        if (nc < minNei) minNei = nc;
        if (nc > maxNei) maxNei = nc;
        if (rtSize == N - 1) fullyConverged++;
    }

    if (neighborCounts.empty()) {
        minNei = 0.0;
        maxNei = 0.0;
    }

    double avgNei = ComputeMean(neighborCounts);
    double stdNei = ComputeStdDev(neighborCounts, avgNei);
    double avgTwo = ComputeMean(twoHopCounts);
    double stdTwo = ComputeStdDev(twoHopCounts, avgTwo);
    double avgRt  = ComputeMean(rtSizes);
    double stdRt  = ComputeStdDev(rtSizes, avgRt);

    // Is node 1 a 1-hop neighbor of node 0 (the victim)?
    // This directly predicts the neighbor_abort rejection reason.
    int node1IsNeighbor = 0;
    if (N >= 2) {
        Ptr<RoutingProtocol> rp1 = cont->Get(1)->GetObject<RoutingProtocol>();
        if (rp1 && rp1->isItNeighbor(Ipv4Address("10.0.0.1"))) {
            node1IsNeighbor = 1;
        }
    }

    // Average per-node minimum Euclidean distance to any other node.
    // Pure physical metric, independent of OLSR convergence.
    double sumMinDist = 0.0;
    uint32_t counted = 0;
    for (uint32_t i = 0; i < N; ++i) {
        Ptr<MobilityModel> mm1 = cont->Get(i)->GetObject<MobilityModel>();
        if (!mm1) continue;
        Vector p1 = mm1->GetPosition();
        double localMin = std::numeric_limits<double>::infinity();
        for (uint32_t j = 0; j < N; ++j) {
            if (i == j) continue;
            Ptr<MobilityModel> mm2 = cont->Get(j)->GetObject<MobilityModel>();
            if (!mm2) continue;
            Vector p2 = mm2->GetPosition();
            double dx = p1.x - p2.x;
            double dy = p1.y - p2.y;
            double dz = p1.z - p2.z;
            double d = std::sqrt(dx*dx + dy*dy + dz*dz);
            if (d < localMin) localMin = d;
        }
        if (std::isfinite(localMin)) {
            sumMinDist += localMin;
            counted++;
        }
    }
    double avgMinDist = (counted > 0) ? (sumMinDist / counted) : 0.0;

    std::ostringstream row;
    row << std::fixed << std::setprecision(6);
    row << RngSeedManager::GetRun() << ","
        << g_currentRun << ","
        << (g_probeMobility ? 1 : 0) << ","
        << avgNei << "," << stdNei << "," << minNei << "," << maxNei << ","
        << avgTwo << "," << stdTwo << ","
        << avgRt  << "," << stdRt  << ","
        << fullyConverged << ","
        << node1IsNeighbor << ","
        << avgMinDist << "\n";

    WriteTopologyProbeRow(row.str());
}

// C++11 compatible logging functions (replace lambda functions)
static void LogDefenseActivation() {
    NS_LOG_INFO("Defense activation phase at " << Simulator::Now().GetSeconds() << "s");
}

static void LogAttackActivation() {
    NS_LOG_INFO("Attack activation phase at " << Simulator::Now().GetSeconds() << "s");
}

// Enhanced simulation statistics with ML dataset info
static void PrintSimStatsWithMLInfo(NodeContainer* cont) {
    std::cout << "@@   New Simulation   @@" << std::endl;
    std::cout << "Seed RngRun: " << RngSeedManager::GetRun() << std::endl;
    std::cout << "ML Dataset Run: " << g_currentRun << std::endl;
    std::cout << "Output Directory: " << g_outputBaseDir << std::endl;
    
	std::cout << "Operational phases:" << std::endl;

	std::cout << "  Baseline (" 
			  << BASELINE_START << "-" << BASELINE_END 
			  << "): Attack OFF, Defence OFF" << std::endl;

	std::cout << "  Attack_only (" 
			  << ATTACK_ONLY_START << "-" << ATTACK_ONLY_END 
			  << "): Attack ON, Defence OFF" << std::endl;

	std::cout << "  Defense_only (" 
			  << DEFENSE_ONLY_START << "-" << DEFENSE_ONLY_END 
			  << "): Attack OFF, Defence ON" << std::endl;

	std::cout << "  Defense_vs_attack (" 
			  << DEFENSE_ATTACK_START << "-" << DEFENSE_ATTACK_END 
			  << "): Attack ON, Defence ON" << std::endl;
    
	std::cout << "Timeline:" << std::endl;
	std::cout << "  0-60     : Initial stabilization" << std::endl;
	std::cout << "  60-100   : baseline" << std::endl;
	std::cout << "  100-160  : attack_only stabilization" << std::endl;
	std::cout << "  160-200  : attack_only" << std::endl;
	std::cout << "  200-260  : defense_only stabilization" << std::endl;
	std::cout << "  260-300  : defense_only" << std::endl;
	std::cout << "  300-360  : defense_vs_attack stabilization" << std::endl;
	std::cout << "  360-400  : defense_vs_attack" << std::endl;
}

/*
static void PrintTopologySet(NodeContainer* cont)
{
	ns3::iolsr::RoutingProtocol rp;
	for (unsigned int i=0;i<cont->GetN();i++)
	{
		Ptr<RoutingProtocol> pt = cont->Get(i)->GetObject<RoutingProtocol>();
		NS_LOG_INFO("Node id: "<<i<<"   "<<"Time: "<<Simulator::Now().GetSeconds());
		pt->PrintTopologySet();
	} 
}
*/
//alternative for cpp simulations:
int main2 (int argc, char *argv[]){

	//CPP version:
	if (__cplusplus == 201703L) std::cout << "C++17\n";
	    else if (__cplusplus == 201402L) std::cout << "C++14\n";
	    else if (__cplusplus == 201103L) std::cout << "C++11\n";
	    else if (__cplusplus == 199711L) std::cout << "C++98\n";
	    else std::cout << "pre-standard C++\n";


	return 0;
}


// --- mobility-model selection (R#3.2, 23/8/2026) ---------------------------
// 'randomwalk2d' is the default and reproduces the campaign's model exactly;
// 'gaussmarkov' gives temporally correlated velocity and direction, which is
// the qualitatively different kind of domain shift Reviewer 3 asked about.
// Mean speed is matched to the RandomWalk2d setting (1.5-2.0 m/s) on purpose,
// so the two domains differ in the STRUCTURE of the motion and not in how fast
// nodes move. Pitch is pinned to zero, keeping the scenario two-dimensional.
static void
ConfigureMobilityModel (MobilityHelper &helper, const std::string &model,
                        double maxX, double maxY)
{
	if (model == "gaussmarkov") {
		helper.SetMobilityModel ("ns3::GaussMarkovMobilityModel",
			"Bounds", BoxValue (Box (0, maxX, 0, maxY, 0, 0)),
			"TimeStep", TimeValue (Seconds (3.0)),
			// Alpha is the fraction of velocity/direction retained per TimeStep, i.e.
			// the memory of the process (ns-3's own default is 1.0 = fully linear).
			// At TimeStep = 3 s, alpha = 0.85 gives a correlation decay time of
			// -3/ln(0.85) ~= 18 s, so a node holds roughly its heading for ~30 m at
			// the configured 1.5-2.0 m/s. This is a declared modelling choice, not a
			// fitted value, and it must be stated in the manuscript (R#3.2).
			"Alpha", DoubleValue (0.85),
			"MeanVelocity", StringValue ("ns3::UniformRandomVariable[Min=1.5|Max=2.0]"),
			"MeanDirection", StringValue ("ns3::UniformRandomVariable[Min=0.0|Max=6.283185307]"),
			"MeanPitch", StringValue ("ns3::ConstantRandomVariable[Constant=0.0]"),
			"NormalVelocity", StringValue ("ns3::NormalRandomVariable[Mean=0.0|Variance=0.04|Bound=0.4]"),
			"NormalDirection", StringValue ("ns3::NormalRandomVariable[Mean=0.0|Variance=0.2|Bound=0.8]"),
			// NormalPitch is type-checked as a NormalRandomVariable (not a plain
			// RandomVariableStream like MeanPitch), so pitch is pinned at zero with a
			// zero-variance normal rather than a ConstantRandomVariable.
			"NormalPitch", StringValue ("ns3::NormalRandomVariable[Mean=0.0|Variance=0.0|Bound=0.0]"));
	} else {
		helper.SetMobilityModel ("ns3::RandomWalk2dMobilityModel",
			"Bounds", RectangleValue (Rectangle (0, maxX, 0, maxY)),
			//"Speed", StringValue("ns3::UniformRandomVariable[Min=0|Max=2.0]"),
			"Speed", StringValue("ns3::UniformRandomVariable[Min=1.5|Max=2.0]"),
			//"Speed", StringValue("ns3::UniformRandomVariable[Min=9.5|Max=10.0]"),
			"Time", TimeValue(Seconds(3.0)),
			"Mode", EnumValue(RandomWalk2dMobilityModel::MODE_TIME));
	}
}

int main (int argc, char *argv[]){
	auto wallStart = std::chrono::steady_clock::now();
	//CPP version:
	//if (__cplusplus == 201703L) std::cout << "C++17\n";
	//    else if (__cplusplus == 201402L) std::cout << "C++14\n";
	//    else if (__cplusplus == 201103L) std::cout << "C++11\n";
	//    else if (__cplusplus == 199711L) std::cout << "C++98\n";
	//    else std::cout << "pre-standard C++\n";

	// Time
	Time::SetResolution (Time::NS);
	
	// Variables
	uint32_t nNodes = 50; //Number of nodes in the simulation, default 50

	//X,Y simulation rectangle - the range of movement of the nodes in the simulation
	double dMaxGridX = 750.0; //default 500x500
	uint32_t nMaxGridX = 750;
	double dMaxGridY = 1000.0;
	uint32_t nMaxGridY = 1000;

	bool bMobility = false; //Delcares whenever there is movement in the network
	std::string mobilityModel = "randomwalk2d"; //R#3.2: 'randomwalk2d' (default) or 'gaussmarkov'
	uint32_t nProtocol = 0; //IOLSR=0, DSDV=1

	// Set running time
	double dSimulationSeconds = 402.0;
	uint32_t nSimulationSeconds = 402;
	//double dSimulationSeconds = 301.0;
	//uint32_t nSimulationSeconds = 301;

	std::string mProtocolName = "Invalid";
	bool bPrintSimStats = true; //simulation stats, such as random seed used, time.
	bool bSuperTransmission = false; //Transmission boost to node X?
	bool bPrintAll = false; //Print routing table for all hops
	bool bPrintFakeCount = false; //Print amount of fake nodes required
	bool bPrintMprFraction = false; //Print fraction of MPR
	bool bPrintRiskyFraction = false; //Print fraction of risky
	bool bIsolationAttack = false; //Execute isolation attack by a node
	bool bEnableAnimation = false;
	bool bEnablePcap = false;
	bIsolationAttackBug = false; //Have an attacker stick to it's target  *****
	bIsolationAttackNeighbor = true; //Execute isolation attack by a random neighbor
	bEnableFictive = false; //Activate fictive defence mode   *****
	bEnableFictiveMitigation = true; //Activate new fictive defence mode (new algorithm, mitigation)
	bool bHighRange = false; //Higher wifi range. 250m should suffice. txGain at 12.4
	// Admission test at t=60. "full" (default) = the paper's AssertConnectivity
	// (every node's table complete); "scenario" = AssertScenarioViable (victim
	// has a route to the source AND >=1 neighbour). Only the realistic-channel
	// experiment sets "scenario" (STATE §25.36); every existing campaign leaves
	// the default, so its behaviour is byte-identical.
	std::string admission = "full";
	bool bPrintTcPowerLevel = false; //Print average TC size
	bool bNeighborDump = false; //Neighbor dump
	bool bIsolationAttackMassive = false; //Execute isolation attack by many nodes
	bool bBlackholeAttack = false; //Replace isolation attack with ported student-DCFM black-hole (spoofing + ANSN poison + data drop)
	uint32_t nSpoofedLinks = 5; //Number of real distant nodes the black-hole spoofs per HELLO/TC
	uint32_t nAttackerNode = 2; //Fixed black-hole attacker node id, placed at grid centre (student default = 2)
	double attackerJitter = 25.0; //Jitter (m) around the grid centre for the attacker position (student default = 25)
	bool bConnectivityPrecentage = true; //Print connectivity precetage every X seconds
	bool bUdpServer = true; //Try to send UDP packets from node1 to node0
	bool bAssertConnectivity = false; //Stop simulation if network is not fully connected, at certain time.
	bool printTotalMprs = false; //prints the MPR sub-network total MPRs, and addresses of MPRs.
	bool printDetectionInC6 = false;
	bool print2hop = false;
	bool printNodesDeclaringFictive = false;
	uint32_t printNodeOutputLog = 999; // 999 to disable this function, otherwise supply node ID.
	uint32_t reportStatsAtTime = 399;   // Final stats reporting


	// Parameters from command line
	CommandLine cmd;
	cmd.AddValue("nNodes", "Number of nodes in the simulation", nNodes);
	cmd.AddValue("nMaxGridX", "X of the simulation rectangle", nMaxGridX);
	cmd.AddValue("nMaxGridY", "Y of the simulation rectangle", nMaxGridY);
	cmd.AddValue("bMobility", "Delcares whenever there is movement in the network", bMobility);
	cmd.AddValue("mobilityModel", "Mobility model used when bMobility=1: 'randomwalk2d' "
		"(default, the campaign's model) or 'gaussmarkov' (temporally correlated "
		"velocity/direction). Ignored when bMobility=0.", mobilityModel);
	cmd.AddValue("nProtocol", "IOLSR=0, DSDV=1", nProtocol);
	cmd.AddValue("nSimulationSeconds", "Amount of seconds to run the simulation", nSimulationSeconds);
	cmd.AddValue("bSuperTransmission", "Transmission boost to node X?", bSuperTransmission);
	cmd.AddValue("bPrintAll", "Print routing table for all hops", bPrintAll);
	cmd.AddValue("bPrintFakeCount", "Print amount of fake nodes required", bPrintFakeCount);
	cmd.AddValue("bPrintMprFraction", "Print fraction of MPR", bPrintMprFraction);
	cmd.AddValue("bPrintRiskyFraction", "Print fraction of risky", bPrintRiskyFraction);
	cmd.AddValue("bPrintTcPowerLevel", "Print average TC size", bPrintTcPowerLevel);
	cmd.AddValue("bIsolationAttack", "Execute isolation attack by a node", bIsolationAttack);
	cmd.AddValue("bIsolationAttackNeighbor", "Execute isolation attack by a random neighbor", bIsolationAttackNeighbor);
	cmd.AddValue("bIsolationAttackMassive", "Execute isolation attack by many nodes", bIsolationAttackMassive);
	cmd.AddValue("bBlackholeAttack", "Replace the isolation attack with the ported student-DCFM black-hole attack", bBlackholeAttack);
	cmd.AddValue("nSpoofedLinks", "Number of real distant nodes the black-hole spoofs per HELLO/TC", nSpoofedLinks);
	cmd.AddValue("nAttackerNode", "Fixed black-hole attacker node id (placed at grid centre)", nAttackerNode);
	cmd.AddValue("attackerJitter", "Jitter (m) around grid centre for the attacker position", attackerJitter);
	cmd.AddValue("bEnableFictive", "Activate fictive defence mode", bEnableFictive);
	cmd.AddValue("propagationModel", "'range' (default, the paper's 190 m disk) or "
	             "'realistic' (range guard + LogDistance + Nakagami; STATE §21.4/§25.26)",
	             g_propModel);
	cmd.AddValue("propRefLoss", "ReferenceLoss (dB @1 m) for the realistic model's "
	             "LogDistance stage — THE calibration knob", g_propRefLoss);
	cmd.AddValue("propExponent", "LogDistance path-loss exponent (realistic model)",
	             g_propExponent);
	cmd.AddValue("propMaxRange", "Hard guard range in metres (realistic model)",
	             g_propMaxRange);
	cmd.AddValue("rtsCtsThreshold", "PSDU size (bytes) above which RTS/CTS is used. "
	             "Default is the ns-3 maximum, i.e. RTS/CTS never fires, which is "
	             "the behaviour of every campaign so far. Set low (e.g. 100) to "
	             "suppress hidden-terminal collisions (R#5.3)",
	             g_rtsCtsThreshold);
	cmd.AddValue("formationLeadIn", "Seconds of pure network formation prepended before "
	             "the four-slot schedule (default 0 = the paper's timeline). Use 60 for "
	             "the window-order experiment so slot 0 gets a stabilisation period like "
	             "every other slot; BOTH arms must use the same value", g_formationLeadIn);
	cmd.AddValue("scenarioOrder", "Window-order experiment: '' = fixed order (default), "
	             "'random' = per-run permutation derived from RngRun, or an explicit "
	             "permutation like '2,0,3,1' (phase indices: 0=baseline 1=attack_only "
	             "2=defense_only 3=defense_vs_attack)", g_scenarioOrder);
	cmd.AddValue("bEnableFictiveMitigation", "Activate new fictive defence mode (new algorithm, mitigation)", bEnableFictiveMitigation);
	cmd.AddValue("bEnableAnimation", "Enable NetAnim animation output", bEnableAnimation);
	cmd.AddValue("bEnablePcap", "Enable PCAP output", bEnablePcap);
	cmd.AddValue("bHighRange", "Higher wifi range", bHighRange);
	cmd.AddValue("admission", "t=60 admission test: 'full' (default, the paper's "
	             "all-nodes-converged check) or 'scenario' (victim has a route to "
	             "the source AND >=1 neighbour; for the realistic-channel run)", admission);
	cmd.AddValue("bNeighborDump", "Neighbor dump", bNeighborDump);
	cmd.AddValue("bConnectivityPrecentage", "Print connectivity precetage every X seconds", bConnectivityPrecentage);
	cmd.AddValue("bIsolationAttackBug", "Have an attacker stick to it's target", bIsolationAttackBug);
	cmd.AddValue("bUdpServer", "Try to send UDP packets from node1 to node0", bUdpServer);
	cmd.AddValue("run", "Run number for output files", g_currentRun);
	cmd.AddValue("outputDir", "Base output directory", g_outputBaseDir);
	cmd.AddValue("enforceHopFilter", "1 (default): reject runs whose UDP source is <3 hops from the victim (paper criterion). 0: admit them and record HOP_CATEGORY on stdout", g_enforceHopFilter);
	cmd.AddValue("windowSeconds", "Measurement-window length in seconds (default 40). "
		"Must leave (windowSeconds - 4) divisible by 2. The UDP probe rate is fixed "
		"(one 512-byte datagram / 2 s, first at +4 s), so the per-window datagram "
		"count follows: (windowSeconds - 4) / 2. All window boundaries and the "
		"simulation end are rederived. See RecomputeWindowTiming().", g_windowSeconds);
	cmd.AddValue("udpInterval", "Seconds between UDP probe datagrams (default 2). "
		"Denser traffic = smaller value. The per-window datagram count is rederived "
		"as (windowSeconds - 4) / udpInterval and must come out whole; enforced "
		"after cmd.Parse. R#5.7 traffic-density experiment.", UDP_PACKET_INTERVAL);
	cmd.AddValue("udpPacketSize", "UDP probe datagram size in bytes (default 512). "
		"Does not affect window timing. R#5.7 traffic-density experiment.", g_udpPacketSize);
	cmd.AddValue("defenseActiveFraction", "Fraction (0,1] of each defended "
		"measurement window during which the defense is active, from the window "
		"start (default 1.0 = whole window, the campaign behaviour). "
		"R#3.11 intermittent-activation experiment.", g_defenseActiveFraction);
	cmd.AddValue("topologyProbeFile", "Path to topology probe CSV (empty = disabled)", g_topologyProbeFile);
	cmd.Parse (argc, argv);

	// --- mobility-model knob (R#3.2, 23/8/2026): reject unknown values loudly,
	// rather than silently falling back to the default and producing a campaign
	// that looks like Gauss-Markov but is not.
	if (mobilityModel != "randomwalk2d" && mobilityModel != "gaussmarkov"){
		std::cout << "*** Invalid --mobilityModel=" << mobilityModel
		          << " (expected 'randomwalk2d' or 'gaussmarkov'). Terminated." << std::endl;
		return 1;
	}
	if (bMobility && mobilityModel != "randomwalk2d"){
		std::cout << "Mobility model: " << mobilityModel << std::endl;
	}

	// --- window-length sweep (13/8/2026): validate the knob, rederive timing ---
	// Reject values that would truncate the probe train (fractional packet) or
	// leave no room for even one probe. This runs BEFORE any Simulator::Schedule,
	// so every schedule below uses the rederived boundaries.
	if (g_windowSeconds < UDP_START_OFFSET_IN_WINDOW + UDP_PACKET_INTERVAL ||
	    std::fmod(g_windowSeconds - UDP_START_OFFSET_IN_WINDOW, UDP_PACKET_INTERVAL) != 0.0){
		std::cout << "*** Invalid --windowSeconds=" << g_windowSeconds
		          << " (need >= UDP_START_OFFSET_IN_WINDOW + udpInterval, and "
		          << "(windowSeconds-4) divisible by udpInterval=" << UDP_PACKET_INTERVAL
		          << "). Terminated." << std::endl;
		return 1;
	}
	RecomputeWindowTiming();
	if (g_defenseActiveFraction <= 0.0 || g_defenseActiveFraction > 1.0){
		std::cout << "*** Invalid --defenseActiveFraction=" << g_defenseActiveFraction
		          << " (need 0 < f <= 1). Terminated." << std::endl;
		return 1;
	}
	if (g_defenseActiveFraction < 1.0 && !g_scenarioOrder.empty()){
		std::cout << "*** --defenseActiveFraction is not wired into the "
		          << "--scenarioOrder path. Terminated." << std::endl;
		return 1;
	}
	if (g_defenseActiveFraction < 1.0){
		std::cout << "Defense active fraction: " << g_defenseActiveFraction
		          << " (deactivation at windowStart + "
		          << g_defenseActiveFraction * MEASUREMENT_DURATION
		          << " s in each defended window)" << std::endl;
	}
	// nSimulationSeconds' default (402) assumes the 40 s window. If the user set
	// a window but not an explicit longer runtime, extend the run to cover the
	// rederived schedule (last event: AbortIfNotReceivedPackets at END+2).
	if (nSimulationSeconds < SIMULATION_END + 3){
		nSimulationSeconds = (uint32_t)(SIMULATION_END + 3);
	}
	if (g_windowSeconds != 40.0){
		std::cout << "Window sweep: windowSeconds=" << g_windowSeconds
		          << " packetsPerWindow=" << UDP_PACKETS_PER_WINDOW
		          << " simulationEnd=" << SIMULATION_END << std::endl;
	}

	// Mirror bMobility to the probe global so RecordTopologyProbe can tag rows
	g_probeMobility = bMobility;

	CreateOutputDirectories();

	if (nSimulationSeconds > 10.0) dSimulationSeconds = nSimulationSeconds; // Force minimum time
	if (nMaxGridX > 10.0) dMaxGridX = nMaxGridX; // Force minimum size. Revert to default.
	if (nMaxGridY > 10.0) dMaxGridY = nMaxGridY; // Force minimum size. Revert to default.

	// Build network
	NodeContainer nodes;
	nodes.Create (nNodes);

	// Add wifi
	WifiHelper wifi;
	// ns-3.47: WifiHelper now defaults to 802.11ax; the 3.19 campaign used the
	// 3.19 default, which was 802.11a (wifi-helper.cc:60). Set it explicitly so
	// the PHY layer matches the campaign instead of silently changing the physics.
	wifi.SetStandard(WIFI_STANDARD_80211a);
	//wifi.SetStandard(WIFI_PHY_STANDARD_80211g);
	// ns-3.47: build the channel explicitly rather than ::Default(). ::Default()
	// prepends a LogDistancePropagationLossModel, and the campaign then chains a
	// RangePropagationLossModel on top. On 3.19 that worked; on 3.47 the stricter
	// PHY error model drops the log-distance-attenuated frame below threshold, so
	// every node sees zero neighbours. The campaign's intent was binary range-based
	// connectivity (that is why RangePropagationLossModel is there), so we use only
	// that model plus the constant-speed delay -- matching the working 3.47
	// reference scenario. Physics is range-binary and deterministic in either ns-3.
	YansWifiChannelHelper wifiChannel;
	wifiChannel.SetPropagationDelay ("ns3::ConstantSpeedPropagationDelayModel");
	YansWifiPhyHelper wifiPhy;              // ns-3.47: no ::Default(), default-construct
	WifiMacHelper wifiMac;                  // ns-3.47: NqosWifiMacHelper removed, use WifiMacHelper
	if (g_propModel == "realistic"){
		// STATE §21.4 recipe: the range model stays as a hard topology guard;
		// the real physics inside it is log-distance + Nakagami fading.
		// Losses in a Yans chain add in dB, so a frame must both be inside
		// the guard range AND survive the faded path loss.
		wifiPhy.Set("TxGain", DoubleValue(12.4));
		wifiChannel.AddPropagationLoss ("ns3::RangePropagationLossModel",
			"MaxRange", DoubleValue (g_propMaxRange));
		wifiChannel.AddPropagationLoss ("ns3::LogDistancePropagationLossModel",
			"Exponent", DoubleValue (g_propExponent),
			"ReferenceDistance", DoubleValue (1.0),
			"ReferenceLoss", DoubleValue (g_propRefLoss));
		wifiChannel.AddPropagationLoss ("ns3::NakagamiPropagationLossModel");
	} else if (bHighRange){
		wifiPhy.Set("TxGain", DoubleValue(12.4));
		wifiChannel.AddPropagationLoss ("ns3::RangePropagationLossModel", "MaxRange", DoubleValue (250));
	} else {
		wifiPhy.Set("TxGain", DoubleValue(12.4));
		wifiChannel.AddPropagationLoss ("ns3::RangePropagationLossModel", "MaxRange", DoubleValue (190));
	}
	wifiPhy.SetChannel(wifiChannel.Create());
	wifi.SetRemoteStationManager ("ns3::ConstantRateWifiManager",
	                             "RtsCtsThreshold",
	                             UintegerValue (g_rtsCtsThreshold));
	wifiMac.SetType ("ns3::AdhocWifiMac");
	//wifiChannel.AddPropagationLoss ("ns3::RangePropagationLossModel", "MaxRange", DoubleValue (105));
	NetDeviceContainer adhocDevices = wifi.Install (wifiPhy, wifiMac, nodes);

	// Rig Node for huge wifi boost
	if (bSuperTransmission){
		NodeContainer superNodes;
		superNodes.Create (1);
		wifiPhy.Set("RxGain", DoubleValue(500.0));
		wifiPhy.Set("TxGain", DoubleValue(500.0));
		NetDeviceContainer superDevices = wifi.Install (wifiPhy, wifiMac, superNodes);
		adhocDevices.Add(superDevices);
		nodes.Add(superNodes);
	}
	
	macTxPerNode.resize(nodes.GetN(), 0);
	macDropPerNode.resize(nodes.GetN(), 0);

	// Install IOLSR / DSDV
	IOlsrHelper iolsr;
	DsdvHelper dsdv;
	Ipv4ListRoutingHelper routeList;
	InternetStackHelper internet;
	std::stringstream tmpStringStream;
	std::string fName = "Stable_Network_Stream";
	fName += "_n";
	tmpStringStream << nNodes;
	fName += tmpStringStream.str();
	tmpStringStream.str("");
	fName += "_x";
	tmpStringStream << nMaxGridX;
	fName += tmpStringStream.str();
	tmpStringStream.str("");
	fName += "_y";
	tmpStringStream << nMaxGridY;
	fName += tmpStringStream.str();
	tmpStringStream.str("");
	fName += "_r";
	tmpStringStream << RngSeedManager::GetRun();
	fName += tmpStringStream.str();
	tmpStringStream.str("");
	fName += ".txt";
	//Ptr<OutputStreamWrapper> stream = Create<OutputStreamWrapper>("Stable_Network_Stream_Run",std::ios::out);	
	//Ptr<OutputStreamWrapper> stream = Create<OutputStreamWrapper>(fName,std::ios::out);	

	switch (nProtocol){
		case 0:
			routeList.Add (iolsr, 100);
			if (!bPrintAll){
				//iolsr.PrintRoutingTableEvery(Seconds(10.0), nodes.Get(1), stream);
			} else {
				Ptr<OutputStreamWrapper> stream = Create<OutputStreamWrapper>(fName,std::ios::out);	
				iolsr.PrintRoutingTableAllEvery(Seconds(10.0), stream);
			}
			break;
		case 1:
			routeList.Add (dsdv, 100);
			if (!bPrintAll) {
				//dsdv.PrintRoutingTableEvery(Seconds(10.0), nodes.Get(1), stream);
			} else {
				Ptr<OutputStreamWrapper> stream = Create<OutputStreamWrapper>(fName,std::ios::out);	
				dsdv.PrintRoutingTableAllEvery(Seconds(10.0), stream);
			}
			break;
		default:
			NS_FATAL_ERROR ("Invalid routing protocol chosen " << nProtocol);
			break;
	}
	internet.SetRoutingHelper(routeList);
	internet.Install (nodes);

	// Install IP
	Ipv4AddressHelper addresses;
	addresses.SetBase ("10.0.0.0", "255.0.0.0");
	Ipv4InterfaceContainer interfaces;
	interfaces = addresses.Assign (adhocDevices);

	// Install mobility
	MobilityHelper mobility;
	Ptr<UniformRandomVariable> randomGridX = CreateObject<UniformRandomVariable> ();
	Ptr<UniformRandomVariable> randomGridY = CreateObject<UniformRandomVariable> ();
	randomGridX->SetAttribute ("Min", DoubleValue (0));
	randomGridX->SetAttribute ("Max", DoubleValue (dMaxGridX));
	randomGridY->SetAttribute ("Min", DoubleValue (0));
	randomGridY->SetAttribute ("Max", DoubleValue (dMaxGridY));

	if (bBlackholeAttack) {
		// BLACK-HOLE placement aligned with the student DCFM harness: the fixed
		// attacker node (nAttackerNode) is pinned at the GRID CENTRE (+/- attackerJitter)
		// with a constant position even under mobility, giving it high betweenness so
		// it actually lies on data paths. All other nodes keep the usual random
		// placement/movement. Gated by bBlackholeAttack -> node-isolation runs unchanged.
		Ptr<UniformRandomVariable> rngJ = CreateObject<UniformRandomVariable> ();
		rngJ->SetAttribute ("Min", DoubleValue (-attackerJitter));
		rngJ->SetAttribute ("Max", DoubleValue (attackerJitter));
		double centerX = dMaxGridX / 2.0;
		double centerY = dMaxGridY / 2.0;
		NodeContainer attackerNodes, normalNodes;
		Ptr<ListPositionAllocator> attackerAlloc = CreateObject<ListPositionAllocator> ();
		Ptr<ListPositionAllocator> normalAlloc = CreateObject<ListPositionAllocator> ();
		for (uint32_t i = 0; i < nodes.GetN (); ++i) {
			if (i == nAttackerNode) {
				double ax = centerX + rngJ->GetValue ();
				double ay = centerY + rngJ->GetValue ();
				if (ax < 0) ax = 0; else if (ax > dMaxGridX) ax = dMaxGridX;
				if (ay < 0) ay = 0; else if (ay > dMaxGridY) ay = dMaxGridY;
				attackerAlloc->Add (Vector (ax, ay, 0.0));
				attackerNodes.Add (nodes.Get (i));
			} else {
				normalAlloc->Add (Vector (randomGridX->GetValue (), randomGridY->GetValue (), 0.0));
				normalNodes.Add (nodes.Get (i));
			}
		}
		MobilityHelper attackerMobility;
		attackerMobility.SetPositionAllocator (attackerAlloc);
		attackerMobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
		attackerMobility.Install (attackerNodes);
		MobilityHelper normalMobility;
		normalMobility.SetPositionAllocator (normalAlloc);
		if (bMobility) {
			ConfigureMobilityModel (normalMobility, mobilityModel, dMaxGridX, dMaxGridY);
		} else {
			normalMobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
		}
		normalMobility.Install (normalNodes);
	} else {
		Ptr<RandomRectanglePositionAllocator> taPositionAlloc = CreateObject<RandomRectanglePositionAllocator> ();
		taPositionAlloc->SetX(randomGridX);
		taPositionAlloc->SetY(randomGridY);
		mobility.SetPositionAllocator (taPositionAlloc);
		if (bMobility) {
			ConfigureMobilityModel (mobility, mobilityModel, dMaxGridX, dMaxGridY);
		} else {
			mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
		}
		mobility.Install (nodes);
	}

	Ptr<UdpServer> udpServer = nullptr;
	if (bUdpServer){
		UdpServerHelper udpServerHelper(80);
		ApplicationContainer apps = udpServerHelper.Install(nodes.Get(0));
		udpServer = DynamicCast<UdpServer>(apps.Get(0));  // ns-3.47: no GetServer(), cast from the installed app
		g_udpServer = udpServer; // expose to SaveScenarioMetrics for per-window UDP metric
		UdpClientHelper udpClientHelper(Ipv4Address("10.0.0.1"), 80);
		udpClientHelper.SetAttribute("Interval", TimeValue(Seconds(UDP_PACKET_INTERVAL))); //The time to wait between packets
		// Cap = the derived per-window count (was a literal 18, which silently
		// truncated the probe train for windows longer than 40 s). This runs
		// after cmd.Parse + RecomputeWindowTiming, so the global is final here.
		udpClientHelper.SetAttribute("MaxPackets", UintegerValue(UDP_PACKETS_PER_WINDOW));
		udpClientHelper.SetAttribute("PacketSize", UintegerValue(g_udpPacketSize));

		
		// Choose random node to become sender
		//Ptr<UniformRandomVariable> rnd = CreateObject<UniformRandomVariable> ();
		//rnd->SetAttribute ("Min", DoubleValue (3));
		//rnd->SetAttribute ("Max", DoubleValue (nodes.GetN()-1));
		

		// Install UdpClient on said node
		//uint32_t sendingNode = rnd->GetInteger();
		uint32_t sendingNode = 1;

		// Baseline window
		{
			ApplicationContainer apps;
			apps.Add(udpClientHelper.Install(nodes.Get(sendingNode)));
			double startTime = BASELINE_START + UDP_START_OFFSET_IN_WINDOW;
			double stopTime = BASELINE_END;
			apps.Start(Seconds(startTime));
			apps.Stop(Seconds(stopTime));
			Simulator::Schedule(Seconds(startTime - 2), &AbortOnNeighbor, nodes.Get(sendingNode), Ipv4Address("10.0.0.1"));
		}

		// Attack-only window
		{
			ApplicationContainer apps;
			apps.Add(udpClientHelper.Install(nodes.Get(sendingNode)));
			double startTime = ATTACK_ONLY_START + UDP_START_OFFSET_IN_WINDOW;
			double stopTime = ATTACK_ONLY_END;
			apps.Start(Seconds(startTime));
			apps.Stop(Seconds(stopTime));
			Simulator::Schedule(Seconds(startTime - 2), &AbortOnNeighbor, nodes.Get(sendingNode), Ipv4Address("10.0.0.1"));
		}

		// Defense-only window
		{
			ApplicationContainer apps;
			apps.Add(udpClientHelper.Install(nodes.Get(sendingNode)));
			double startTime = DEFENSE_ONLY_START + UDP_START_OFFSET_IN_WINDOW;
			double stopTime = DEFENSE_ONLY_END;
			apps.Start(Seconds(startTime));
			apps.Stop(Seconds(stopTime));
			Simulator::Schedule(Seconds(startTime - 2), &AbortOnNeighbor, nodes.Get(sendingNode), Ipv4Address("10.0.0.1"));
		}

		// Defense+Attack window
		{
			ApplicationContainer apps;
			apps.Add(udpClientHelper.Install(nodes.Get(sendingNode)));
			double startTime = DEFENSE_ATTACK_START + UDP_START_OFFSET_IN_WINDOW;
			double stopTime = DEFENSE_ATTACK_END;
			apps.Start(Seconds(startTime));
			apps.Stop(Seconds(stopTime));
			Simulator::Schedule(Seconds(startTime - 2), &AbortOnNeighbor, nodes.Get(sendingNode), Ipv4Address("10.0.0.1"));
		}

		Simulator::Schedule(Seconds(SIMULATION_END + 2), &AbortIfNotReceivedPackets, udpServer);
		Simulator::Schedule(Seconds(SIMULATION_END - 1), &ReportNumReceivedPackets, udpServer);	
	}


	if (bPrintSimStats) {
	    Simulator::Schedule(Seconds(5), &PrintSimStatsWithMLInfo, &nodes);
	}
	
	if(printDetectionInC6){
		Simulator::Schedule(Seconds (reportStatsAtTime+0.02), &PrintC6Detection, &nodes);
	}

	if (bPrintFakeCount) {
	// Execute PrintCountFakeNodes after the delay time
		//Ptr<OutputStreamWrapper> wrap = Create<OutputStreamWrapper>("StableNetwork-Mod-FakeNodes.txt", ios::out);
		Simulator::Schedule(Seconds (reportStatsAtTime), &PrintCountFakeNodes, &nodes);
		//Simulator::Schedule(Seconds (240), &PrintCountFakeNodes, &nodes);
	}

	if(printNodeOutputLog != 999){
		Simulator::Schedule(Seconds (reportStatsAtTime+0.3), &PrintNodeOutputLog, &nodes, printNodeOutputLog);
	}

	if (printNodesDeclaringFictive) {
	// Execute PrintCountFakeNodes after the delay time
		//Ptr<OutputStreamWrapper> wrap = Create<OutputStreamWrapper>("StableNetwork-Mod-FakeNodes.txt", ios::out);
		Simulator::Schedule(Seconds (reportStatsAtTime), &PrintNodesDeclaredFictive, &nodes);
		//Simulator::Schedule(Seconds (240), &PrintCountFakeNodes, &nodes);
	}

	if (bPrintMprFraction) {
	// Execute PrintMprFraction after the delay time 
		Simulator::Schedule(Seconds (reportStatsAtTime), &PrintMprFraction, &nodes);
		//Simulator::Schedule(Seconds (240), &PrintMprFraction, &nodes);
	}

	if(printTotalMprs){
		Simulator::Schedule(Seconds (reportStatsAtTime), &PrintMprs, &nodes);

	}

	if(print2hop){
		Simulator::Schedule(Seconds (reportStatsAtTime), &Print2hopNeighborsOfVictim, &nodes);
	}

	if (bPrintRiskyFraction) {
	// Execute PrintMprFraction after the delay time 
		Ipv4Address ignore = Ipv4Address("0.0.0.0");
		if (bIsolationAttackBug) ignore = Ipv4Address("10.0.0.3"); //If an attacker stick to it's target then we ignore 10.0.0.3
		Simulator::Schedule(Seconds (reportStatsAtTime), &PrintRiskyFraction, &nodes, ignore);
		//Simulator::Schedule(Seconds (240), &PrintRiskyFraction, &nodes, ignore);
	}
	if (bPrintTcPowerLevel) {
	// If bPrintTcPowerLevel -> Print average TC size
		Simulator::Schedule(Seconds (reportStatsAtTime), &PrintTcPowerLevel, &nodes);
		//Simulator::Schedule(Seconds (240), &PrintTcPowerLevel, &nodes);
	}

	// ------------------------------------------------------------
	// Connectivity check after initial stabilization
	// ------------------------------------------------------------
	// Topology probe runs at t=59s for every seed (before AssertConnectivity at
	// t=60s), so rejected runs also contribute rows to the probe CSV. See
	// RecordTopologyProbe() for details.
	Simulator::Schedule(Seconds(59.0), &RecordTopologyProbe, &nodes);
	// Admission test. Default 'full' schedules the paper's AssertConnectivity,
	// unchanged; 'scenario' swaps in the viability check for the realistic run.
	if (admission == "scenario")
		Simulator::Schedule(Seconds(INITIAL_STABILIZATION), &AssertScenarioViable, &nodes);
	else
		Simulator::Schedule(Seconds(INITIAL_STABILIZATION), &AssertConnectivity, &nodes);
	Simulator::Schedule(Seconds(INITIAL_STABILIZATION), &CheckAndReportConnectivity, &nodes);

	// ------------------------------------------------------------
	// Four operational phases in the same run:
	// baseline -> attack_only -> defense_only -> defense_vs_attack
	// (or, with --scenarioOrder, the same four phases in a permuted
	//  slot order — STATE §23.10 / §25.16)
	// ------------------------------------------------------------

	if (!g_scenarioOrder.empty()) {
		// ---- permuted path (window-order experiment) ----
		uint32_t perm[4] = {0, 1, 2, 3};
		if (g_scenarioOrder == "random") {
			// Deterministic Fisher-Yates from RngRun (LCG; no extra headers).
			// Same run number -> same permutation, independent of the ns-3
			// RNG streams, so enabling this cannot perturb the simulation's
			// own randomness.
			uint64_t s = 0x9E3779B97F4A7C15ull ^ (uint64_t) RngSeedManager::GetRun();
			for (uint32_t i = 3; i > 0; --i) {
				s = s * 6364136223846793005ull + 1442695040888963407ull;
				uint32_t j = (uint32_t)((s >> 33) % (uint64_t)(i + 1));
				uint32_t t = perm[i]; perm[i] = perm[j]; perm[j] = t;
			}
		} else {
			std::stringstream ss(g_scenarioOrder);
			std::string tok;
			uint32_t n = 0;
			bool seen[4] = {false, false, false, false};
			while (std::getline(ss, tok, ',') && n < 4) {
				int v = std::atoi(tok.c_str());
				if (v < 0 || v > 3) {
					std::cerr << "ERROR: scenarioOrder token out of range: " << tok << std::endl;
					return 1;
				}
				perm[n++] = (uint32_t) v;
				seen[v] = true;
			}
			if (n != 4 || !(seen[0] && seen[1] && seen[2] && seen[3])) {
				std::cerr << "ERROR: scenarioOrder must be 'random' or a permutation "
				          << "of 0,1,2,3 — got: " << g_scenarioOrder << std::endl;
				return 1;
			}
		}
		// One greppable line per run; with StartTime in the CSVs this makes
		// the slot assignment fully recoverable offline.
		std::cout << "ScenarioOrder: run=" << RngSeedManager::GetRun()
		          << " slots=[" << PHASES[perm[0]].label << ","
		          << PHASES[perm[1]].label << "," << PHASES[perm[2]].label
		          << "," << PHASES[perm[3]].label << "]" << std::endl;

		// Slot 0's config time is the END of the formation period. With
		// --formationLeadIn=60 that is t=60, giving slot 0 the same 60 s of
		// settled-with-config stabilisation (60->120) that slots 1-3 get.
		// With the default lead-in of 0 this is t=0 — today's behaviour, under
		// which an attack-bearing slot 0 aborts and the seed is rejected.
		double stabStart[4] = {g_formationLeadIn, ATTACK_ONLY_STABILIZATION_START,
		                       DEFENSE_ONLY_STABILIZATION_START,
		                       DEFENSE_ATTACK_STABILIZATION_START};
		double measStart[4] = {BASELINE_START, ATTACK_ONLY_START,
		                       DEFENSE_ONLY_START, DEFENSE_ATTACK_START};
		for (uint32_t slot = 0; slot < 4; ++slot) {
			const PhaseSpec& ph = PHASES[perm[slot]];
			// Each slot starts from a clean state and applies its phase's
			// config. Deactivate* are idempotent flag-clears, safe when
			// already off. NOTE one deliberate difference from the fixed
			// order: there, the defence stays continuously ON from the
			// defense_only stabilisation through defense_vs_attack (never
			// deactivated mid-run); here every slot is self-contained, so a
			// defence-on slot following a defence-on slot deactivates and
			// reactivates at the boundary.
			Simulator::Schedule(Seconds(stabStart[slot]), &DisableIsolationAttack, &nodes);
			Simulator::Schedule(Seconds(stabStart[slot]), &DisableBlackholeAttack, &nodes);
			Simulator::Schedule(Seconds(stabStart[slot]), &DeactivateFictiveDefence, &nodes);
			Simulator::Schedule(Seconds(stabStart[slot]), &DeactivateFictiveMitigation, &nodes);
			if (ph.attackOn) {
				if (bBlackholeAttack)
					Simulator::Schedule(Seconds(stabStart[slot]), &ExecuteBlackholeAttackOnNode, &nodes, nAttackerNode, nSpoofedLinks);
				else
					Simulator::Schedule(Seconds(stabStart[slot]), &ExecuteIsolationAttackByNeighbor, &nodes, Ipv4Address("10.0.0.1"));
			}
			if (ph.defenseOn) {
				Simulator::Schedule(Seconds(stabStart[slot]), &ActivateFictiveDefence, &nodes);
				Simulator::Schedule(Seconds(stabStart[slot]), &ActivateFictiveMitigation, &nodes);
			}
			Simulator::Schedule(Seconds(measStart[slot]), &StartPhaseMeasurement,
			                    &nodes, udpServer, perm[slot], measStart[slot],
			                    measStart[slot] + MEASUREMENT_DURATION);
		}
	} else {

	// Initial state: no attack, no defense
	Simulator::Schedule(Seconds(0.0), &DisableIsolationAttack, &nodes);
	Simulator::Schedule(Seconds(0.0), &DisableBlackholeAttack, &nodes);

	// --------------------
	// attack_only phase
	// --------------------
	// At 100s start the stabilization period for attack_only:
	// attack ON, defense OFF
	Simulator::Schedule(Seconds(ATTACK_ONLY_STABILIZATION_START), &DisableIsolationAttack, &nodes);
	Simulator::Schedule(Seconds(ATTACK_ONLY_STABILIZATION_START), &DisableBlackholeAttack, &nodes);
	// Select attack type: black-hole (ported student DCFM) or node-isolation (original).
	if (bBlackholeAttack)
		Simulator::Schedule(Seconds(ATTACK_ONLY_STABILIZATION_START), &ExecuteBlackholeAttackOnNode, &nodes, nAttackerNode, nSpoofedLinks);
	else
		Simulator::Schedule(Seconds(ATTACK_ONLY_STABILIZATION_START), &ExecuteIsolationAttackByNeighbor, &nodes, Ipv4Address("10.0.0.1"));

	// --------------------
	// defense_only phase
	// --------------------
	// At 200s start the stabilization period for defense_only:
	// attack OFF, defense ON
	Simulator::Schedule(Seconds(DEFENSE_ONLY_STABILIZATION_START), &DisableIsolationAttack, &nodes);
	Simulator::Schedule(Seconds(DEFENSE_ONLY_STABILIZATION_START), &DisableBlackholeAttack, &nodes);
	Simulator::Schedule(Seconds(DEFENSE_ONLY_STABILIZATION_START), &ActivateFictiveDefence, &nodes);
	Simulator::Schedule(Seconds(DEFENSE_ONLY_STABILIZATION_START), &ActivateFictiveMitigation, &nodes);

	// --------------------
	// defense_vs_attack phase
	// --------------------
	// At 300s start the stabilization period for defense_vs_attack:
	// attack ON, defense ON
	// Select attack type: black-hole (ported student DCFM) or node-isolation (original).
	if (bBlackholeAttack)
		Simulator::Schedule(Seconds(DEFENSE_ATTACK_STABILIZATION_START), &ExecuteBlackholeAttackOnNode, &nodes, nAttackerNode, nSpoofedLinks);
	else
		Simulator::Schedule(Seconds(DEFENSE_ATTACK_STABILIZATION_START), &ExecuteIsolationAttackByNeighbor, &nodes, Ipv4Address("10.0.0.1"));
	Simulator::Schedule(Seconds(DEFENSE_ATTACK_STABILIZATION_START), &ActivateFictiveDefence, &nodes);
	Simulator::Schedule(Seconds(DEFENSE_ATTACK_STABILIZATION_START), &ActivateFictiveMitigation, &nodes);

	// R#3.11: intermittent activation. With fraction < 1 the defense is switched
	// off partway through each DEFENDED window. Stabilization is untouched, so
	// the window opens in the established state; defense_vs_attack re-activates
	// at its own stabilization start (the schedules above), exactly as before.
	// At the default 1.0 nothing is scheduled and the run is bit-identical.
	if (g_defenseActiveFraction < 1.0){
		double defOnFor = g_defenseActiveFraction * MEASUREMENT_DURATION;
		Simulator::Schedule(Seconds(DEFENSE_ONLY_START + defOnFor),   &DeactivateFictiveDefence,    &nodes);
		Simulator::Schedule(Seconds(DEFENSE_ONLY_START + defOnFor),   &DeactivateFictiveMitigation, &nodes);
		Simulator::Schedule(Seconds(DEFENSE_ATTACK_START + defOnFor), &DeactivateFictiveDefence,    &nodes);
		Simulator::Schedule(Seconds(DEFENSE_ATTACK_START + defOnFor), &DeactivateFictiveMitigation, &nodes);
	}

	// ------------------------------------------------------------
	// Measurement windows
	// ------------------------------------------------------------
	Simulator::Schedule(Seconds(BASELINE_START),       &StartBaselineMeasurement,      &nodes, udpServer);
	Simulator::Schedule(Seconds(ATTACK_ONLY_START),    &StartAttackOnlyMeasurement,    &nodes, udpServer);
	Simulator::Schedule(Seconds(DEFENSE_ONLY_START),   &StartDefenseOnlyMeasurement,   &nodes, udpServer);
	Simulator::Schedule(Seconds(DEFENSE_ATTACK_START), &StartDefenseAttackMeasurement, &nodes, udpServer);

	} // end fixed-order path (--scenarioOrder unset)

	// Animation
	if (bEnableAnimation) {
		AnimationInterface anim ("Stable_Network_animation.xml");
		anim.SetMobilityPollInterval (Seconds (1));
	}

	// Pcap
	if (bEnablePcap) {
		wifiPhy.EnablePcap("DelayTest_", NodeContainer(nodes.Get(0)));
		wifiPhy.EnablePcap("DelayTest_", NodeContainer(nodes.Get(1)));
	}
	//wifiPhy.EnablePcap("DelayTest_", nodes.Get(1));
	//wifiPhy.EnablePcapAll ("DelayTest_");
	//std::ostringstream sspcap;
	//sspcap << "BottleNeck_" << "FixPos_" << bFixPos << "_";
	//wifiPhy.EnablePcapAll (sspcap.str());

	g_flowMonitor = g_flowHelper.InstallAll();

	Config::Connect("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Mac/MacTx", MakeCallback(&MacTxCallback));
	Config::Connect("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Mac/MacTxDrop", MakeCallback(&MacTxDropCallback));
	// v4: PHY-level transmit count - the air truth for X_TransmissionRate,
	// including control frames and retransmissions. Pure observation.
	Config::Connect("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/$ns3::YansWifiPhy/PhyTxBegin", MakeCallback(&PhyTxBeginCallback));

	for (uint32_t i = 0; i < nodes.GetN (); ++i)
  	{
    	Ptr<Ipv4> ipv4 = nodes.Get (i)->GetObject<Ipv4> ();
    	ipv4->TraceConnectWithoutContext ("Tx", MakeCallback (&TraceOlsrPacket));
  	}

	// --- passive vantage points (revision 2026-08) -------------------------
	SelectObserverNodes (nodes.GetN (), (uint32_t) RngSeedManager::GetRun ());
	std::cout << "Observer nodes:";
	for (size_t oi = 0; oi < g_observerNodes.size (); ++oi)
	  {
	    std::cout << " " << g_observerNodes[oi];
	    std::ostringstream path;
	    path << "/NodeList/" << g_observerNodes[oi]
	         << "/DeviceList/*/$ns3::WifiNetDevice/Phy/$ns3::YansWifiPhy/MonitorSnifferRx";
	    Config::ConnectWithoutContext (path.str (),
	        MakeBoundCallback (&ObserverSniffRx, (uint32_t) oi));
	  }
	std::cout << std::endl;
	// -----------------------------------------------------------------------
	
	// Run simulation
	NS_LOG_INFO ("Run simulation.");
	Simulator::Stop (Seconds (dSimulationSeconds));

	Simulator::Run ();

	//RngSeedManager::GetRun()
	// std::ostringstream oss;
	// oss << "./simulations/features/att-1_def-1/metrics_output-" << RngSeedManager::GetRun() << ".csv";

	// ExtractAndLogMetrics(flowMon, flowHelper, nodes, oss.str().c_str());

	Simulator::Destroy ();

	std::cout << "Simulation finished at sim time "
          << Simulator::Now().GetSeconds()
          << " s" << std::endl;
		  
	auto wallEnd = std::chrono::steady_clock::now();
	double elapsed = std::chrono::duration<double>(wallEnd - wallStart).count();

	std::cout << "Runtime for 4 windows: "
          << elapsed << " seconds" << std::endl;
	return 0;
}