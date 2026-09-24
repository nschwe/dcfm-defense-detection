/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */

#include "iolsr-defense-fpnt.h"
#include "iolsr-routing-protocol.h"
#include "iolsr-repositories.h"
#include "ns3/log.h"
#include "ns3/double.h"
#include "ns3/pointer.h"
#include "ns3/uinteger.h"
#include "ns3/boolean.h"
#include "ns3/node.h"
#include "ns3/ipv4.h"
#include "ns3/ipv4-header.h"
#include "ns3/ipv4-l3-protocol.h"
#include "ns3/ipv4-interface.h"
#include "ns3/arp-cache.h"
#include "ns3/node-list.h"
#include "ns3/wifi-net-device.h"
#include <algorithm>
#include <cmath>
#include <set>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("IolsrDefenseFpnt");

namespace iolsr {

NS_OBJECT_ENSURE_REGISTERED (IolsrDefenseFpnt);

// ============================================================================
// Fuzzy Petri net -- static structure (paper Fig. 2)
// ----------------------------------------------------------------------------
// Place indices (15 places):
//    0  p1   "load is high"                       (evidence)
//    1  p2   "load is low"                        (evidence)
//    2  p3   "PFR is high"                        (evidence)
//    3  p4   "PFR is low"                         (evidence)
//    4  p5   "avg forwarding delay is high"       (evidence)
//    5  p6   "avg forwarding delay is low"        (evidence)
//    6  p7   "routing deviation observed"         (evidence)
//    7  p8   "routing normal"                     (evidence)
//    8  p9   "compromised / serious attacker"     (intermediate)
//    9  p10  "lightly malicious (jellyfish-like)" (intermediate)
//   10  p11  "routing integrity attacker"         (intermediate)
//   11  p12  "data plane normal"                  (intermediate)
//   12  p13  "routing plane normal"               (intermediate)
//   13  p14  "node cannot be trusted"             (verdict: distrust)
//   14  p15  "node can be trusted"                (verdict: trust)
//
// Transition indices (11 transitions -- the paper's own m = 11). The OR rules
// R1 and R6 use one transition per input branch because Type-2 competitive
// semantics threshold-test each input place independently.
//
//    0  R1-a  p1 -> p9         tau=0.4, omega=1.0,          mu=0.9
//    1  R1-b  p4 -> p9         tau=0.4, omega=1.0,          mu=0.9
//    2  R1-c  p5 -> p9         tau=0.5, omega=1.0,          mu=0.6
//    3  R2    p2,p5 -> p10     tau=0.5, omega=(0.6,0.4),     mu=0.8
//    4  R3    p7 -> p11        tau=0.5, omega=1.0,          mu=0.9
//    5  R4    p3,p6,p2 -> p12  tau=0.7, omega=(0.6,0.3,0.1), mu=0.9
//    6  R5    p8 -> p13        tau=0.8, omega=1.0,          mu=1.0
//    7  R6-a  p9 -> p14        tau=0.4, omega=1.0,          mu=0.9
//    8  R6-b  p10 -> p14       tau=0.4, omega=1.0,          mu=0.7
//    9  R6-c  p11 -> p14       tau=0.4, omega=1.0,          mu=0.9
//   10  R7    p12,p13 -> p15   tau=0.6, omega=(0.5,0.5),     mu=0.9
// ============================================================================

namespace {

// Place indices.
constexpr int P1_LOAD_HIGH      = 0;
constexpr int P2_LOAD_LOW       = 1;
constexpr int P3_FWD_HIGH       = 2;
constexpr int P4_FWD_LOW        = 3;
constexpr int P5_DELAY_HIGH     = 4;
constexpr int P6_DELAY_LOW      = 5;
constexpr int P7_ROUTE_BAD      = 6;
constexpr int P8_ROUTE_OK       = 7;
constexpr int P14_DISTRUST      = 13;
constexpr int P15_TRUST         = 14;

constexpr int NUM_PLACES      = 15;
constexpr int NUM_TRANSITIONS = 11;

// Transition firing thresholds (Algorithm 1, Step 2).
// Order: R1a, R1b, R1c, R2, R3, R4, R5, R6a, R6b, R6c, R7.
const std::vector<double> TH = { 0.4, 0.4, 0.5, 0.5, 0.5, 0.7, 0.8, 0.4, 0.4, 0.4, 0.6 };

// Input incidence matrix W^T (transitions x places).
// Row r lists the weighted inputs (omega) of transition r.
const std::vector<std::vector<double>> W_T = {
    // p1   p2   p3   p4   p5   p6   p7   p8   p9   p10  p11  p12  p13  p14  p15
    {  1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }, // R1a: p1
    {  0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }, // R1b: p4
    {  0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }, // R1c: p5
    {  0.0, 0.6, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }, // R2 : p2,p5
    {  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }, // R3 : p7
    {  0.0, 0.1, 0.6, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }, // R4 : p3,p6,p2
    {  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }, // R5 : p8
    {  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }, // R6a: p9
    {  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 }, // R6b: p10
    {  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0 }, // R6c: p11
    {  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0 }, // R7 : p12,p13
};

// Output incidence matrix U (places x transitions).
// U[p][r] = weight (mu) of the arc from transition r to place p.
const std::vector<std::vector<double>> U_MAT = {
    //  R1a  R1b  R1c  R2   R3   R4   R5   R6a  R6b  R6c  R7
    {  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }, // p1   (input only)
    {  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }, // p2   (input only)
    {  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }, // p3   (input only)
    {  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }, // p4   (input only)
    {  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }, // p5   (input only)
    {  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }, // p6   (input only)
    {  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }, // p7   (input only)
    {  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }, // p8   (input only)
    {  0.9, 0.9, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }, // p9   <- R1a,R1b,R1c
    {  0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }, // p10  <- R2
    {  0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }, // p11  <- R3
    {  0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0 }, // p12  <- R4
    {  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0 }, // p13  <- R5
    {  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.7, 0.9, 0.0 }, // p14  <- R6a,R6b,R6c
    {  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9 }, // p15  <- R7
};

// Maximum iterations of the reasoning loop (Algorithm 1, Step 5). The longest
// causal chain is two hops (evidence -> intermediate -> verdict), so under
// monotone updates convergence takes at most two rounds; 10 is a safety guard.
constexpr int MAX_FPN_ITERATIONS = 10;

// Tolerance for detecting the "all recommendations identical" case in the
// pairwise L1 DIF computation.
constexpr double DIF_EPS = 1e-9;

// OLSR_TOP_HOLD_TIME = 3 * TcInterval. D1 fires when the most recent TC from
// an MPR is older than this window.
constexpr double D1_HOLD_MULTIPLIER = 3.0;

/// Fingerprint identifying one IP datagram across hops. TTL and checksum
/// change at every hop and are deliberately excluded; source, destination,
/// identification and payload size do not.
inline uint64_t
HashIpv4Header (const Ipv4Header& h)
{
  uint64_t v = (uint64_t (h.GetSource ().Get ()) << 32)
             | uint64_t (h.GetDestination ().Get ());
  v ^= (uint64_t (h.GetIdentification ()) << 16) | uint64_t (h.GetPayloadSize ());
  v = (v ^ (v >> 30)) * 0xbf58476d1ce4e5b9ULL;
  v = (v ^ (v >> 27)) * 0x94d049bb133111ebULL;
  v = v ^ (v >> 31);
  return v;
}

} // anonymous namespace

// ============================================================================
// TypeId & construction
// ============================================================================

TypeId
IolsrDefenseFpnt::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::iolsr::IolsrDefenseFpnt")
    .SetParent<IolsrDefenseStrategy> ()
    .SetGroupName ("Olsr")
    .AddConstructor<IolsrDefenseFpnt> ()
    .AddAttribute ("TrustUpdateInterval",
                   "Trust evaluation period 't' (Section 5.1). The routing "
                   "protocol schedules PeriodicCheck at exactly this cadence, "
                   "and the load factor is normalized by it.",
                   TimeValue (Seconds (5.0)),
                   MakeTimeAccessor (&IolsrDefenseFpnt::m_checkInterval),
                   MakeTimeChecker ())
    .AddAttribute ("MaliciousThreshold",
                   "Trust value below which a node is reported malicious. "
                   "The paper defines no such threshold -- FPNT-OLSR avoids "
                   "low-trust nodes by path selection rather than isolating "
                   "them (Section 2.3) -- so this only drives logging, the "
                   "GetBlacklist() accessor and the reactive reroute trigger.",
                   DoubleValue (0.2),
                   MakeDoubleAccessor (&IolsrDefenseFpnt::m_maliciousThreshold),
                   MakeDoubleChecker<double> (0.0, 1.0))
    .AddAttribute ("UncertaintyBeta",
                   "Beta in Equation (4): T = E_trust + beta * E_uncertain. "
                   "Larger beta = more optimistic reading of the undetermined "
                   "share of the evidence.",
                   DoubleValue (0.6),
                   MakeDoubleAccessor (&IolsrDefenseFpnt::m_uncertaintyBeta),
                   MakeDoubleChecker<double> (0.0, 1.0))
    .AddAttribute ("HistoryFadingFactor",
                   "Lambda in Equation (5): T = (1-lambda)*T_c + lambda*T_{c-1}.",
                   DoubleValue (0.7),
                   MakeDoubleAccessor (&IolsrDefenseFpnt::m_fadingFactor),
                   MakeDoubleChecker<double> (0.0, 1.0))
    .AddAttribute ("MaxLoad",
                   "Normalization constant X (bits/second) for the NORM "
                   "operator of Definition 10 applied to the load factor. The "
                   "paper calls it the node's load capacity. Load at or above "
                   "this maps to truth degree 1.",
                   DoubleValue (1.0e6),
                   MakeDoubleAccessor (&IolsrDefenseFpnt::m_maxLoad),
                   MakeDoubleChecker<double> (1.0))
    .AddAttribute ("MaxDelay",
                   "Normalization constant (seconds) for the NORM operator "
                   "applied to the average forwarding delay; the paper calls "
                   "it the node's delay tolerance.",
                   DoubleValue (0.5),
                   MakeDoubleAccessor (&IolsrDefenseFpnt::m_maxDelay),
                   MakeDoubleChecker<double> (0.001))
    .AddAttribute ("CheatThreshold",
                   "delta in Section 5.1.D: the protocol-deviation flag p7 is "
                   "raised only once Count_rcheat exceeds this, which the "
                   "paper introduces to absorb false detections caused by "
                   "unreliable wireless links.",
                   UintegerValue (2),
                   MakeUintegerAccessor (&IolsrDefenseFpnt::m_cheatThreshold),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("Enabled",
                   "Runtime toggle. When false, IsMalicious returns false for "
                   "every node, GetBlacklist is empty, IsTrustRoutingEnabled "
                   "is false and PeriodicCheck is a no-op. Every transition "
                   "triggers a full state reset so the defense never carries "
                   "evidence across a disabled window.",
                   BooleanValue (true),
                   MakeBooleanAccessor (&IolsrDefenseFpnt::SetEnabled,
                                        &IolsrDefenseFpnt::GetEnabled),
                   MakeBooleanChecker ())

    // ---------------------------------------------------------------------
    // Beyond the paper. Each default reproduces the paper's own behavior.
    // ---------------------------------------------------------------------
    .AddAttribute ("StickyEvidence",
                   "Carry a factor's truth degree over from the previous "
                   "period when the current period produced no observation "
                   "for it. NOT in the paper: Section 5.1 states the counters "
                   "are cleared every t, and Equation (5) is where history is "
                   "supposed to enter. Turning this on makes a neighbor "
                   "unable to reset its evaluation by falling silent, at the "
                   "cost of weighting history twice.",
                   BooleanValue (false),
                   MakeBooleanAccessor (&IolsrDefenseFpnt::m_stickyEvidence),
                   MakeBooleanChecker ())
    .AddAttribute ("DemoteUnverifiedNodes",
                   "Return a below-beta trust value for a node that appears "
                   "only in received advertisements and in no OLSR state of "
                   "our own. NOT in the paper, which gives an unevaluated "
                   "node E_trust=0, E_uncertain=1 and hence T = beta. Turning "
                   "this on penalizes link-spoofing phantoms.",
                   BooleanValue (false),
                   MakeBooleanAccessor (&IolsrDefenseFpnt::m_demoteUnverified),
                   MakeBooleanChecker ())
    .AddAttribute ("RollbackOnMacFailure",
                   "Decrement Count_rcv when our own MAC layer reports that a "
                   "frame to the neighbor exhausted its retry limit. NOT in "
                   "the paper. Turning this on stops our own link failures "
                   "from depressing the neighbor's packet forwarding rate.",
                   BooleanValue (false),
                   MakeBooleanAccessor (&IolsrDefenseFpnt::m_rollbackOnMacFailure),
                   MakeBooleanChecker ())
    .AddAttribute ("MonitorTcForwarding",
                   "Include the TC term of the packet forwarding rate "
                   "(Section 5.1.B: a TC arriving at an MPR V_j raises "
                   "Count_rcv, and V_j retransmitting it raises Count_fwd). "
                   "IS in the paper but off by default here, because the "
                   "obligation has to be inferred -- hearing a TC ourselves "
                   "is not proof that V_j heard it -- so a collision at V_j "
                   "is charged to V_j as a routing-plane failure.",
                   BooleanValue (false),
                   MakeBooleanAccessor (&IolsrDefenseFpnt::m_monitorTcForwarding),
                   MakeBooleanChecker ())
    .AddAttribute ("FlatTrustDiagnostic",
                   "FPNT-DIAG (temporary, not in the paper). Make GetNodeTrust "
                   "return the same value for every node, so Algorithm 2 runs "
                   "on a flat metric. Isolates route construction from trust "
                   "value dispersion when defense_only loses traffic.",
                   BooleanValue (false),
                   MakeBooleanAccessor (&IolsrDefenseFpnt::m_flatTrust),
                   MakeBooleanChecker ())
    .AddAttribute ("BroadcastLoad",
                   "Count broadcast and multicast frames toward Count_load, "
                   "per Definition 1 -- \"V_j receives ANY TYPE of packet\". "
                   "The audited fpnt347 reference drops them, because a group "
                   "MAC address cannot be attributed to one selector from the "
                   "frame alone; this reconstructs the attribution from our own "
                   "2-hop and topology state. OLSR's entire control plane is "
                   "broadcast, so with this off only the few nodes carrying "
                   "unicast data have load evidence at all, and R2 tags exactly "
                   "those. Off by default so the reference is reproduced.",
                   BooleanValue (false),
                   MakeBooleanAccessor (&IolsrDefenseFpnt::m_broadcastLoad),
                   MakeBooleanChecker ())
    .AddAttribute ("RetainEvaluations",
                   "When a node stops being one of our MPR selectors, keep the "
                   "computed evaluation vector and the s^(0) history instead of "
                   "erasing them; only the raw per-period counters are pruned. "
                   "Section 5.2 limits what an MPR ADVERTISES to its current "
                   "selectors but does not say to forget, and Equation (5) models "
                   "history explicitly. Matters under mobility, where the selector "
                   "set turns over constantly and evidence never accumulates. Off "
                   "by default: it also increases trust dispersion, which is what "
                   "costs delivery (STATE 25.85.2).",
                   BooleanValue (false),
                   MakeBooleanAccessor (&IolsrDefenseFpnt::m_retainEvaluations),
                   MakeBooleanChecker ())
  ;
  return tid;
}

IolsrDefenseFpnt::IolsrDefenseFpnt ()
  : m_protocol (nullptr),
    m_selfAddress (),
    m_checkInterval (Seconds (5.0)),
    m_maliciousThreshold (0.2),
    m_uncertaintyBeta (0.6),
    m_fadingFactor (0.7),
    m_maxLoad (1.0e6),
    m_maxDelay (0.5),
    m_cheatThreshold (2),
    m_enabled (true),
    m_stickyEvidence (false),
    m_demoteUnverified (false),
    m_rollbackOnMacFailure (false),
    m_monitorTcForwarding (false),
    m_flatTrust (false),
    m_broadcastLoad (false),
    m_retainEvaluations (false)
{
  NS_LOG_FUNCTION (this);
}

IolsrDefenseFpnt::~IolsrDefenseFpnt ()
{
  NS_LOG_FUNCTION (this);
}

void
IolsrDefenseFpnt::Setup (RoutingProtocol* proto, Ipv4Address nodeAddress)
{
  NS_LOG_FUNCTION (this << nodeAddress);
  m_protocol = proto;
  m_selfAddress = nodeAddress;
  NS_LOG_INFO ("FPNT-OLSR initialized for node " << nodeAddress);
}

void
IolsrDefenseFpnt::DoDispose ()
{
  NS_LOG_FUNCTION (this);
  m_metrics.clear ();
  m_deviationCounts.clear ();
  m_trustTable.clear ();
  m_directEvaluationVectors.clear ();
  m_lastSValues.clear ();
  m_recommendations.clear ();
  m_pendingArrivals.clear ();
  m_lastTcTime.clear ();
  m_mprSelectionTime.clear ();
  m_seenTcForD2.clear ();
  m_pendingTcRelays.clear ();
  m_protocol = nullptr;
  // IolsrDefenseStrategy::DoDispose is pure virtual in this tree, so chain
  // straight to the ns-3 Object implementation.
  Object::DoDispose ();
}

void
IolsrDefenseFpnt::SetEnabled (bool enabled)
{
  NS_LOG_FUNCTION (this << enabled);

  const bool wasEnabled = m_enabled;
  m_enabled = enabled;

  // Cold-start reset on EVERY state transition. Both directions wipe the
  // accumulated state so neither phase inherits residue from the other: an
  // attack -> defense transition must not contaminate the first evaluation
  // with pre-enable metrics, and an off -> on cycle must not accumulate
  // across phases. No-op calls (same value twice) are skipped.
  if (wasEnabled == enabled)
    {
      return;
    }

  NS_LOG_INFO ("FPNT-OLSR " << (enabled ? "enabled" : "disabled")
               << " on " << m_selfAddress
               << " - clearing all accumulated state for cold start");

  m_metrics.clear ();
  m_deviationCounts.clear ();
  m_pendingArrivals.clear ();
  m_recommendations.clear ();
  m_directEvaluationVectors.clear ();
  m_lastSValues.clear ();
  m_trustTable.clear ();
  m_seenTcForD2.clear ();
  m_pendingTcRelays.clear ();

  // D1/D2 bookkeeping. Without clearing m_lastTcTime, a long disabled window
  // followed by an enable would make every MPR look silent (last TC seen
  // "long ago") and trip D1 on the very first scan.
  m_lastTcTime.clear ();
  m_mprSelectionTime.clear ();

  // Recompute now, on the empty trust state. Which algorithm that runs
  // depends on the new value of m_enabled: trust routing while enabled,
  // stock RFC 3626 once disabled. Without this the pre-transition routing
  // table survives until the next HELLO or TC triggers a recompute.
  if (m_protocol)
    {
      m_protocol->RecomputeRoutingTable ();
    }
}

bool
IolsrDefenseFpnt::GetEnabled () const
{
  return m_enabled;
}

// ============================================================================
// Trust query methods
// ============================================================================

bool
IolsrDefenseFpnt::IsMalicious (Ipv4Address addr)
{
  if (!m_enabled)
    {
      return false;
    }
  auto it = m_trustTable.find (addr);
  if (it == m_trustTable.end ())
    {
      // No verdict until direct traffic or a recommendation produces one.
      return false;
    }
  return (it->second < m_maliciousThreshold);
}

std::set<Ipv4Address>
IolsrDefenseFpnt::GetBlacklist () const
{
  std::set<Ipv4Address> blacklist;
  if (!m_enabled)
    {
      return blacklist;
    }
  for (const auto& entry : m_trustTable)
    {
      if (entry.second < m_maliciousThreshold)
        {
          blacklist.insert (entry.first);
        }
    }
  return blacklist;
}

std::vector<EvaluationVector>
IolsrDefenseFpnt::GetEvaluationVectors (const std::vector<Ipv4Address>& neighbors)
{
  NS_LOG_FUNCTION (this);
  if (!m_enabled)
    {
      return {};
    }

  // Section 5.2: piggyback our direct evaluations of our MPR selectors onto
  // the TC we are about to flood. The TC already advertises exactly those
  // addresses, so one vector per address costs no extra control message.
  std::vector<EvaluationVector> vectors;
  vectors.reserve (neighbors.size ());

  for (const auto& addr : neighbors)
    {
      auto it = m_directEvaluationVectors.find (addr);
      if (it != m_directEvaluationVectors.end ())
        {
          vectors.push_back (it->second);
        }
      else
        {
          // No observation yet. Encode "no opinion" as full uncertainty
          // (E_trust = 0, E_distrust = 0, E_uncertain = 1), which is exactly
          // what Algorithm 1 Step 6 produces from an all-zero token vector.
          EvaluationVector ev;
          ev.trust     = 0;
          ev.distrust  = 0;
          ev.uncertain = 255;
          vectors.push_back (ev);
        }
    }

  return vectors;
}

double
IolsrDefenseFpnt::GetNodeTrust (Ipv4Address node)
{
  if (!m_enabled)
    {
      return m_uncertaintyBeta;
    }
  if (m_flatTrust)
    {
      return m_uncertaintyBeta; // FPNT-DIAG: flat metric, no node preferred.
    }
  auto it = m_trustTable.find (node);
  if (it != m_trustTable.end ())
    {
      return it->second;
    }

  if (m_demoteUnverified && m_protocol != nullptr)
    {
      bool seenInRealState = false;
      for (const auto& nb : m_protocol->GetNeighbors ())
        {
          if (nb.neighborMainAddr == node)
            {
              seenInRealState = true;
              break;
            }
        }
      if (!seenInRealState)
        {
          for (const auto& nb2 : m_protocol->GetTwoHopNeighbors ())
            {
              if (nb2.twoHopNeighborAddr == node || nb2.neighborMainAddr == node)
                {
                  seenInRealState = true;
                  break;
                }
            }
        }
      if (!seenInRealState)
        {
          // Known only from someone else's advertisement. Kept above the
          // malicious threshold so it stays routable when nothing else is,
          // but below beta so any node with even one observation wins.
          return std::max (m_maliciousThreshold + 0.01, m_uncertaintyBeta * 0.5);
        }
    }

  // No evidence at all: Algorithm 1 on an empty token vector yields
  // E_trust = 0, E_uncertain = 1, so Equation (4) gives T = beta.
  return m_uncertaintyBeta;
}

IolsrDefenseFpnt::DebugStateSizes
IolsrDefenseFpnt::GetDebugStateSizes () const
{
  DebugStateSizes s;
  s.metrics                 = m_metrics.size ();
  s.trustTable              = m_trustTable.size ();
  s.directEvaluationVectors = m_directEvaluationVectors.size ();
  s.lastSValues             = m_lastSValues.size ();
  s.recommendations         = m_recommendations.size ();
  s.pendingArrivals         = m_pendingArrivals.size ();
  s.lastTcTime              = m_lastTcTime.size ();
  s.mprSelectionTime        = m_mprSelectionTime.size ();
  s.blacklist               = GetBlacklist ().size ();
  return s;
}

// ============================================================================
// Small helpers
// ============================================================================

std::set<Ipv4Address>
IolsrDefenseFpnt::SelectorsInRadiusOf (Ipv4Address transmitter) const
{
  std::set<Ipv4Address> out;
  if (m_protocol == nullptr || transmitter == Ipv4Address::GetAny ())
    {
      return out;
    }

  // Everything our link state says is one hop from the transmitter.
  std::set<Ipv4Address> inRadius;
  for (const auto& nb2 : m_protocol->GetTwoHopNeighbors ())
    {
      if (nb2.neighborMainAddr == transmitter) inRadius.insert (nb2.twoHopNeighborAddr);
      if (nb2.twoHopNeighborAddr == transmitter) inRadius.insert (nb2.neighborMainAddr);
    }
  for (const auto& topo : m_protocol->m_state.GetTopologySet ())
    {
      if (topo.lastAddr == transmitter) inRadius.insert (topo.destAddr);
      if (topo.destAddr == transmitter) inRadius.insert (topo.lastAddr);
    }

  for (const auto& sel : m_protocol->GetMprSelectors ())
    {
      const Ipv4Address vj = sel.mainAddr;
      if (vj != transmitter && inRadius.find (vj) != inRadius.end ())
        {
          out.insert (vj);
        }
    }
  return out;
}

bool
IolsrDefenseFpnt::IsOurMprSelector (Ipv4Address addr) const
{
  if (m_protocol == nullptr)
    {
      return false;
    }
  // Paper Section 5.1: "only MPRs are responsible for monitoring and
  // evaluating their selectors". GetMprSelectors() -- not GetMprSet() --
  // is the right direction: selectors are the nodes that chose US as their
  // MPR, which is the relationship the paper puts us in charge of.
  for (const auto& sel : m_protocol->GetMprSelectors ())
    {
      if (sel.mainAddr == addr)
        {
          return true;
        }
    }
  return false;
}

Time
IolsrDefenseFpnt::GetTopologyHoldTime () const
{
  if (m_protocol == nullptr)
    {
      return Seconds (0);
    }
  TimeValue tcInterval;
  m_protocol->GetAttribute ("TcInterval", tcInterval);
  return tcInterval.Get () * D1_HOLD_MULTIPLIER;
}

// ============================================================================
// Recommendation intake and deviation rule D2 (TC omission)
// ============================================================================

void
IolsrDefenseFpnt::OnRecvEvaluationVectors (
    Ipv4Address sender,
    const std::vector<Ipv4Address>& advertisedNeighbors,
    const std::vector<EvaluationVector>& vectors)
{
  NS_LOG_FUNCTION (this << sender);

  if (advertisedNeighbors.size () != vectors.size ())
    {
      NS_LOG_WARN ("Malformed TC from " << sender << ": size mismatch ("
                   << advertisedNeighbors.size () << " vs "
                   << vectors.size () << ")");
      return;
    }

  for (size_t i = 0; i < advertisedNeighbors.size (); ++i)
    {
      const Ipv4Address target = advertisedNeighbors[i];
      // Keyed by (originator, target): a later TC from the same originator
      // about the same target replaces the earlier one. Counting both would
      // let a chatty MPR weight itself up in Equations (1)-(3).
      m_recommendations[{sender, target}] = vectors[i];

      NS_LOG_DEBUG ("Recommendation from " << sender << " about " << target
                    << ": T=" << static_cast<int> (vectors[i].trust)
                    << " D=" << static_cast<int> (vectors[i].distrust)
                    << " U=" << static_cast<int> (vectors[i].uncertain));
    }
}

void
IolsrDefenseFpnt::OnRecvHello (Ipv4Address, Ptr<const Packet>,
                              const MessageHeader&, const MessageHeader::Hello&)
{
  // Not used by the paper's model.
}

void
IolsrDefenseFpnt::OnRecvTc (Ipv4Address senderIfaceAddr, Ptr<const Packet>,
                           const MessageHeader& msg, const MessageHeader::Tc& tc)
{
  NS_LOG_FUNCTION (this << senderIfaceAddr);

  if (m_protocol == nullptr)
    {
      return;
    }

  const Ipv4Address originator = msg.GetOriginatorAddress ();
  const uint16_t msgSeq = msg.GetMessageSequenceNumber ();
  const Ipv4Address forwarder = m_protocol->GetMainAddress (senderIfaceAddr);
  const Time now = Simulator::Now ();

  // D1 bookkeeping: most recent TC time per originator.
  m_lastTcTime[originator] = now;

  // ----------------------------------------------------------------------
  // Optional TC term of the packet forwarding rate (Section 5.1.B).
  //
  // Numerator: this copy reaching us from a forwarder other than the
  // originator IS a retransmission by that forwarder.
  // Denominator: a TC we hear will also be heard by our other neighbors, so
  // each MPR selector of ours that itself originates TCs (hence is an MPR,
  // hence must relay) acquires an obligation for this (originator, seq).
  // ----------------------------------------------------------------------
  if (m_monitorTcForwarding && m_enabled)
    {
      const std::pair<Ipv4Address, uint16_t> tcId {originator, msgSeq};

      if (forwarder != originator && IsOurMprSelector (forwarder))
        {
          auto pit = m_pendingTcRelays.find (forwarder);
          if (pit != m_pendingTcRelays.end () && pit->second.erase (tcId) > 0)
            {
              m_metrics[forwarder].countFwd++;
              NS_LOG_LOGIC ("Observed " << forwarder << " relaying TC from " << originator);
            }
        }

      for (const auto& sel : m_protocol->GetMprSelectors ())
        {
          const Ipv4Address vj = sel.mainAddr;
          if (vj == forwarder || vj == originator)
            {
              continue;
            }
          // "V_j is a MPR node": only a node with selectors of its own
          // originates TCs, so having seen a TC from V_j is our observable
          // proxy for that condition.
          if (m_lastTcTime.find (vj) == m_lastTcTime.end ())
            {
              continue;
            }
          auto& pending = m_pendingTcRelays[vj];
          if (pending.insert ({tcId, now}).second)
            {
              m_metrics[vj].countRcv++;
            }
        }
    }

  // ----------------------------------------------------------------------
  // Deviation rule D2 (Section 5.1.D, first case): an MPR of ours that sends
  // a TC omitting our address is suspected of isolating us.
  // ----------------------------------------------------------------------
  const MprSet mprs = m_protocol->GetMprSet ();
  if (mprs.find (originator) == mprs.end ())
    {
      // Not our MPR -- it owes us no advertisement.
      return;
    }

  // The routing protocol delivers this hook for every copy of a flooded TC,
  // duplicates included. Judging each copy would multiply a single omission
  // by the number of relays that reach us and blow straight past delta.
  if (!m_seenTcForD2.insert ({originator, msgSeq}).second)
    {
      return;
    }

  // Maintain the shared first-observation map so D1 and D2 use one definition
  // of the grace window.
  auto sit = m_mprSelectionTime.find (originator);
  if (sit == m_mprSelectionTime.end ())
    {
      m_mprSelectionTime[originator] = now;
      return; // first time we see them as our MPR -- grant grace.
    }

  // D2 stays suppressed during the grace window: they may not have learned we
  // exist yet (HELLO exchange still in progress).
  if (now - sit->second < GetTopologyHoldTime ())
    {
      return;
    }

  const auto& advertised = tc.neighborAddresses;
  const bool selfIsAdvertised =
      std::find (advertised.begin (), advertised.end (), m_selfAddress) != advertised.end ();

  if (!selfIsAdvertised)
    {
      m_deviationCounts[originator]++;
      NS_LOG_LOGIC ("Deviation D2: MPR " << originator << " omitted us from TC");
    }
}

void
IolsrDefenseFpnt::OnTcGenerated (const MessageHeader::Tc&)
{
  // Not used by the paper's model.
}

// ============================================================================
// Metric collection hooks -- Section 5.1
// ============================================================================

void
IolsrDefenseFpnt::OnDataPacketReceived (Ptr<const Packet> /*packet*/,
                                       Ipv4Address /*source*/,
                                       Ipv4Address /*destination*/,
                                       Ipv4Address /*nextHop*/)
{
  // Fires on us as we are about to forward, at the same moment as
  // OnDataPacketForwarded, which has the IP header we need. Everything is
  // accumulated there instead.
}

void
IolsrDefenseFpnt::OnDataPacketForwarded (Ptr<const Packet> /*packet*/,
                                        Ipv4Address /*nextHop*/,
                                        Ipv4Address /*finalDest*/)
{
  // Header-less variant. FPNT needs the IPv4 header to fingerprint the packet,
  // so all accounting happens in the four-argument override below; the routing
  // protocol always calls that one.
}

void
IolsrDefenseFpnt::OnDataPacketForwarded (const Ipv4Header& header,
                                        Ptr<const Packet> packet,
                                        Ipv4Address nextHop,
                                        Ipv4Address finalDest)
{
  if (nextHop.IsBroadcast () || nextHop.IsAny ())
    {
      return;
    }
  if (!IsOurMprSelector (nextHop))
    {
      return; // V_j did not select us as its MPR -- outside our scope.
    }

  auto& metrics = m_metrics[nextHop];

  // Definition 1 (Load). Count the datagram as it appears on the wire,
  // header included, so this path and the promiscuous path below agree.
  metrics.countLoad += packet->GetSize () + header.GetSerializedSize ();

  // Definition 2, denominator: only packets V_j is expected to forward,
  // per the paper's "DATA.dest != V_j" clause.
  if (nextHop != finalDest)
    {
      metrics.countRcv++;

      auto& pending = m_pendingArrivals[nextHop];
      const uint64_t key = HashIpv4Header (header);
      // Never overwrite an existing seed: a retransmission of the same hop
      // must not reset the arrival time, or the measured delay is biased
      // toward zero.
      if (pending.find (key) == pending.end ())
        {
          pending[key] = { Simulator::Now () };
        }
      NS_LOG_LOGIC ("Seeded arrival at " << nextHop << " fp=" << std::hex << key << std::dec
                    << " id=" << header.GetIdentification ()
                    << " payload=" << header.GetPayloadSize ());
    }
}

void
IolsrDefenseFpnt::OnDataPacketDropped (Ptr<const Packet>, Ipv4Address,
                                      Ipv4Address, DropReason)
{
  // Drops show up implicitly in the ratio countFwd / countRcv.
}

void
IolsrDefenseFpnt::OnNeighborForwardedPacket (Mac48Address transmitter,
                                            Mac48Address receiver,
                                            Ptr<const Packet> packet)
{
  if (m_protocol == nullptr)
    {
      return;
    }

  Ipv4Header ipHeader;
  if (packet->PeekHeader (ipHeader) == 0)
    {
      return; // not an IPv4 packet -- out of scope
    }
  const uint64_t fingerprint = HashIpv4Header (ipHeader);

  // ====================================================================
  // Transmitter side: V_j --(DATA)--> *
  //   Section 5.1.B numerator, and the departure half of the Definition 3
  //   delay measurement.
  // ====================================================================
  const Ipv4Address txAddr = MacToIpv4 (transmitter);
  if (txAddr != Ipv4Address::GetAny () && IsOurMprSelector (txAddr))
    {
      auto& pending = m_pendingArrivals[txAddr];
      NS_LOG_LOGIC ("Sniffed tx by " << txAddr << " fp=" << std::hex << fingerprint << std::dec
                    << " id=" << ipHeader.GetIdentification ()
                    << " payload=" << ipHeader.GetPayloadSize ()
                    << " pending=" << pending.size ()
                    << " match=" << (pending.count (fingerprint) ? "YES" : "no"));
      auto it = pending.find (fingerprint);
      if (it != pending.end ())
        {
          auto& metrics = m_metrics[txAddr];
          metrics.countFwd++;
          const double delay = (Simulator::Now () - it->second.arrivalTime).GetSeconds ();
          if (delay >= 0.0)
            {
              metrics.totalDelay += delay;
            }
          pending.erase (it);
          NS_LOG_LOGIC ("Observed " << txAddr << " forwarding packet (fingerprint "
                        << std::hex << fingerprint << std::dec << ")");
        }
    }

  // ====================================================================
  // Receiver side: * --DATA--> V_j
  //   Section 5.1.A (load) and 5.1.B (PFR denominator) both quantify over
  //   ANY transmitter, not just us. Without this arm a blackhole sitting on
  //   a path that no monitoring MPR happens to feed would show PFR = 0/0 --
  //   no evidence -- and never be flagged.
  //
  //   Our own transmissions never reach here: MonitorSnifferRx in the
  //   routing protocol drops frames whose transmitter is one of our MACs,
  //   because OnDataPacketForwarded already counted them.
  // ====================================================================
  if (receiver.IsGroup ())
    {
      // Definition 1 counts EVERY packet V_j receives, and OLSR's whole
      // control plane is broadcast. The group MAC address names no single
      // V_j, so the attribution is reconstructed: a frame from this
      // transmitter reached every node inside its radius, which our own
      // 2-hop and topology state can enumerate. Load only -- the PFR
      // denominator keeps its separate gating.
      if (m_broadcastLoad)
        {
          const uint32_t bytes = packet->GetSize ();
          for (const Ipv4Address& vj : SelectorsInRadiusOf (txAddr))
            {
              m_metrics[vj].countLoad += bytes;
            }
        }
      return; // no per-V_j receiver to charge the forwarding rate to
    }

  const Ipv4Address rxAddr = MacToIpv4 (receiver);
  if (rxAddr == Ipv4Address::GetAny () || !IsOurMprSelector (rxAddr))
    {
      return;
    }

  auto& rxMetrics = m_metrics[rxAddr];

  // Definition 1 (Load): every packet V_j receives, whether or not V_j is
  // the final destination -- Section 5.1.A gates on nothing.
  rxMetrics.countLoad += packet->GetSize ();

  // Definition 2 (PFR denominator): only packets V_j must forward.
  if (ipHeader.GetDestination () != rxAddr)
    {
      rxMetrics.countRcv++;

      auto& pending = m_pendingArrivals[rxAddr];
      if (pending.find (fingerprint) == pending.end ())
        {
          pending[fingerprint] = { Simulator::Now () };
        }

      NS_LOG_LOGIC ("Observed packet --> " << rxAddr << " (fingerprint "
                    << std::hex << fingerprint << std::dec << ")");
    }
}

// ----------------------------------------------------------------------------
// Hooks outside the paper's model.
// ----------------------------------------------------------------------------

void IolsrDefenseFpnt::OnQueueStatusReport (uint32_t, uint32_t) {}

void IolsrDefenseFpnt::OnEnergyStateUpdate (double, double) {}

void
IolsrDefenseFpnt::OnMacTxFailure (Ipv4Address neighbor, uint32_t count)
{
  // OnDataPacketForwarded fires at intent-to-send, before the MAC layer has
  // had a go, so countRcv is already up. If the MAC then exhausts its retry
  // limit the packet never reached V_j, and charging it to V_j understates
  // its forwarding rate. The paper does not model this, so the correction is
  // opt-in (RollbackOnMacFailure).
  if (!m_enabled || !m_rollbackOnMacFailure)
    {
      return;
    }
  if (neighbor == Ipv4Address::GetAny () || neighbor.IsBroadcast ())
    {
      return;
    }

  auto it = m_metrics.find (neighbor);
  if (it == m_metrics.end ())
    {
      return;
    }

  // countLoad is left alone: the callback does not carry the packet size, and
  // the bias is far below the resolution of the load factor's NORM.
  it->second.countRcv = (it->second.countRcv >= count) ? (it->second.countRcv - count) : 0;
}

// ============================================================================
// Trust reasoning cycle -- Algorithm 1 plus Equations (1)-(5)
// ============================================================================

void
IolsrDefenseFpnt::PeriodicCheck ()
{
  NS_LOG_FUNCTION (this);

  if (!m_enabled)
    {
      // Hooks fire regardless of the flag (the routing protocol does not gate
      // them), so drain whatever accumulated or a later enable would act on
      // stale evidence.
      m_metrics.clear ();
      m_deviationCounts.clear ();
      m_recommendations.clear ();
      m_pendingArrivals.clear ();
      m_seenTcForD2.clear ();
      m_pendingTcRelays.clear ();
      return;
    }

  // Drop data-plane state for nodes that are no longer our MPR selectors: the
  // paper scopes monitoring to the current selector set. Deviation counters
  // are NOT in m_metrics and so are not affected -- they are indexed by our
  // MPRs, which are generally not our selectors, and pruning them here would
  // discard every D2 observation before it was ever evaluated.
  if (m_protocol != nullptr)
    {
      std::set<Ipv4Address> currentSelectors;
      for (const auto& s : m_protocol->GetMprSelectors ())
        {
          currentSelectors.insert (s.mainAddr);
        }

      for (auto it = m_metrics.begin (); it != m_metrics.end (); )
        {
          if (currentSelectors.find (it->first) == currentSelectors.end ())
            {
              m_pendingArrivals.erase (it->first);
              m_pendingTcRelays.erase (it->first);
              if (!m_retainEvaluations)
                {
                  // The computed opinion and its evidence history. Erasing
                  // these is what makes a lost selector relationship restart
                  // the evaluation from zero rather than resume it.
                  m_directEvaluationVectors.erase (it->first);
                  m_lastSValues.erase (it->first);
                }
              it = m_metrics.erase (it);
            }
          else
            {
              ++it;
            }
        }
    }

  // ----- Step 0: deviation rule D1 (silent MPRs) and observation cleanup.
  ScanDeviationRuleD1 ();
  ExpireStaleObservations ();

  bool needsReactiveUpdate = false;

  // ----- Step 1: refresh the direct evaluation of every node we have any
  // evidence about -- data-plane metrics, routing-plane deviations, or both.
  std::set<Ipv4Address> evaluated;
  for (const auto& entry : m_metrics)
    {
      evaluated.insert (entry.first);
    }
  for (const auto& entry : m_deviationCounts)
    {
      evaluated.insert (entry.first);
    }

  const NodeBehaviorMetrics emptyMetrics;
  for (const Ipv4Address& addr : evaluated)
    {
      auto mit = m_metrics.find (addr);
      auto dit = m_deviationCounts.find (addr);
      const NodeBehaviorMetrics& metrics = (mit != m_metrics.end ()) ? mit->second : emptyMetrics;
      const uint32_t deviations = (dit != m_deviationCounts.end ()) ? dit->second : 0;

      const std::vector<double> s0 = MetricsToS0 (addr, metrics, deviations);
      m_lastSValues[addr] = s0;
      m_directEvaluationVectors[addr] = RunFuzzyPetriNet (s0);
    }

  // ----- Step 2: the union of targets needing a trust update -- anything we
  // evaluated ourselves, plus anything a recommendation arrived about.
  std::set<Ipv4Address> targets;
  for (const auto& entry : m_directEvaluationVectors)
    {
      targets.insert (entry.first);
    }
  for (const auto& entry : m_recommendations)
    {
      targets.insert (entry.first.second);
    }

  // ----- Step 3: aggregation plus Equations (4)-(5) for every target.
  for (const Ipv4Address& addr : targets)
    {
      // Evidence set: our direct evaluation (if any) plus every received
      // recommendation. The paper gives both the same standing.
      std::vector<EvaluationVector> evs;

      auto itDirect = m_directEvaluationVectors.find (addr);
      if (itDirect != m_directEvaluationVectors.end ())
        {
          evs.push_back (itDirect->second);
        }

      for (const auto& entry : m_recommendations)
        {
          if (entry.first.second == addr)
            {
              evs.push_back (entry.second);
            }
        }

      if (evs.empty ())
        {
          continue;
        }

      // Equations (1)-(3): pairwise L1 aggregation with the slander filter.
      double aggTrust     = 0.0;
      double aggUncertain = 0.0;
      AggregateEvaluations (evs, aggTrust, aggUncertain);

      // Equation (4).
      double Tc = aggTrust + m_uncertaintyBeta * aggUncertain;
      Tc = std::clamp (Tc, 0.0, 1.0);

      // Equation (5), with an explicit first-period bootstrap: T_{c-1} does
      // not exist yet, and blending against an invented value would distort
      // the first verdict.
      auto itOld = m_trustTable.find (addr);
      const bool hasHistory = (itOld != m_trustTable.end ());
      const double finalTrust = hasHistory
          ? ((1.0 - m_fadingFactor) * Tc + m_fadingFactor * itOld->second)
          : Tc;

      const bool wasMalicious = hasHistory && (itOld->second < m_maliciousThreshold);
      const bool isMalicious  = (finalTrust < m_maliciousThreshold);
      if (wasMalicious != isMalicious)
        {
          if (isMalicious)
            {
              NS_LOG_WARN ("Node " << addr << " dropped below the trust threshold (T = "
                           << finalTrust << ")");
            }
          else
            {
              NS_LOG_INFO ("Node " << addr << " recovered above the trust threshold (T = "
                           << finalTrust << ")");
            }
          needsReactiveUpdate = true;
        }

      m_trustTable[addr] = finalTrust;
    }

  // ----- Step 4: clear the per-period state. Section 5.1: "these counters
  // are cleared every t". m_directEvaluationVectors and m_lastSValues persist
  // (they are overwritten when fresh data for the same neighbor arrives);
  // m_pendingArrivals is handled by ExpireStaleObservations.
  m_metrics.clear ();
  m_deviationCounts.clear ();
  m_recommendations.clear ();
  m_seenTcForD2.clear ();

  // ----- Step 5: tell the routing layer if any verdict flipped.
  if (needsReactiveUpdate && m_protocol)
    {
      m_protocol->RecomputeRoutingTable ();
    }


}

// ============================================================================
// Deviation rule D1 (silent-MPR detection)
// ============================================================================

void
IolsrDefenseFpnt::ScanDeviationRuleD1 ()
{
  if (m_protocol == nullptr)
    {
      return;
    }

  const Time holdTime = GetTopologyHoldTime ();
  const MprSet mprs = m_protocol->GetMprSet ();
  const Time now = Simulator::Now ();

  // Reap selection times for nodes that are no longer our MPRs.
  for (auto it = m_mprSelectionTime.begin (); it != m_mprSelectionTime.end (); )
    {
      if (mprs.find (it->first) == mprs.end ())
        {
          it = m_mprSelectionTime.erase (it);
        }
      else
        {
          ++it;
        }
    }

  for (const Ipv4Address& mpr : mprs)
    {
      // A newly selected MPR gets a holdTime grace window; without it, it is
      // certain to be flagged on the very next scan simply because it has not
      // had time to emit a TC yet.
      auto sit = m_mprSelectionTime.find (mpr);
      if (sit == m_mprSelectionTime.end ())
        {
          m_mprSelectionTime[mpr] = now;
          continue;
        }
      const Time selectedSince = sit->second;
      if (now - selectedSince < holdTime)
        {
          continue;
        }

      // Past the grace window: apply the silence test.
      auto it = m_lastTcTime.find (mpr);
      const Time age = (it == m_lastTcTime.end ())
          ? (now - selectedSince)  // never seen a TC: measure from selection,
                                   // not from the start of the simulation
          : (now - it->second);

      if (age > holdTime)
        {
          m_deviationCounts[mpr]++;
          NS_LOG_LOGIC ("Deviation D1: MPR " << mpr << " silent for "
                        << age.GetSeconds () << " s");
        }
    }
}

void
IolsrDefenseFpnt::ExpireStaleObservations ()
{
  // Arrival timestamps older than two trust periods correspond to packets
  // that entered a neighbor and were never seen to leave -- either dropped
  // (which the PFR factor already captures) or missed by our sniffer.
  // Expiring them keeps the maps bounded.
  const Time cutoff = Simulator::Now () - m_checkInterval * 2;

  for (auto& entry : m_pendingArrivals)
    {
      auto& pending = entry.second;
      for (auto it = pending.begin (); it != pending.end (); )
        {
          it = (it->second.arrivalTime < cutoff) ? pending.erase (it) : std::next (it);
        }
    }

  for (auto& entry : m_pendingTcRelays)
    {
      auto& pending = entry.second;
      for (auto it = pending.begin (); it != pending.end (); )
        {
          it = (it->second < cutoff) ? pending.erase (it) : std::next (it);
        }
    }
}

// ============================================================================
// Metric normalization -- Section 5.1 A through D
// ============================================================================

std::vector<double>
IolsrDefenseFpnt::MetricsToS0 (Ipv4Address addr,
                              const NodeBehaviorMetrics& metrics,
                              uint32_t deviationCount)
{
  // S^(0) over the 15 places. Only p1..p8 take evidence; p9..p15 are the
  // intermediate and verdict places the net fills in.
  std::vector<double> S (NUM_PLACES, 0.0);

  const bool hasHistory = m_stickyEvidence &&
                          (m_lastSValues.find (addr) != m_lastSValues.end ());
  static const std::vector<double> zeros (NUM_PLACES, 0.0);
  const std::vector<double>& last = hasHistory ? m_lastSValues.at (addr) : zeros;

  const double t = m_checkInterval.GetSeconds ();
  const double tSafe = (t > 0.0) ? t : 1.0;

  // ---- Factor 1: Load (Definition 1) ----
  // countLoad is in bytes; the paper defines load in bit/s. Convert, then
  // apply NORM against m_maxLoad (also bit/s).
  if (metrics.countLoad > 0)
    {
      const double loadBps = (metrics.countLoad * 8.0) / tSafe;
      const double s1 = std::clamp (loadBps / m_maxLoad, 0.0, 1.0);
      S[P1_LOAD_HIGH] = s1;
      S[P2_LOAD_LOW]  = 1.0 - s1; // REVS, Definition 11
    }
  else if (hasHistory)
    {
      S[P1_LOAD_HIGH] = last[P1_LOAD_HIGH];
      S[P2_LOAD_LOW]  = last[P2_LOAD_LOW];
    }
  // else: no evidence -> (0, 0), which propagates as uncertainty. That is the
  // right reading for a neighbor we have not observed.

  // ---- Factor 2: Packet forwarding rate (Definition 2) ----
  if (metrics.countRcv > 0)
    {
      // Clamped because promiscuous countFwd can pick up a retransmission
      // whose arrival we never saw.
      double pfr = static_cast<double> (metrics.countFwd)
                 / static_cast<double> (metrics.countRcv);
      pfr = std::clamp (pfr, 0.0, 1.0);
      S[P3_FWD_HIGH] = pfr;
      S[P4_FWD_LOW]  = 1.0 - pfr;
    }
  else if (hasHistory)
    {
      S[P3_FWD_HIGH] = last[P3_FWD_HIGH];
      S[P4_FWD_LOW]  = last[P4_FWD_LOW];
    }

  // ---- Factor 3: Average forwarding delay (Definition 3) ----
  if (metrics.countFwd > 0 && metrics.totalDelay > 0.0)
    {
      const double avgDelay = metrics.totalDelay / static_cast<double> (metrics.countFwd);
      const double s5 = std::clamp (avgDelay / m_maxDelay, 0.0, 1.0);
      S[P5_DELAY_HIGH] = s5;
      S[P6_DELAY_LOW]  = 1.0 - s5;
    }
  else if (hasHistory)
    {
      S[P5_DELAY_HIGH] = last[P5_DELAY_HIGH];
      S[P6_DELAY_LOW]  = last[P6_DELAY_LOW];
    }
  // countFwd > 0 with totalDelay == 0 means no matched packet, so no delay
  // evidence -- reporting a spuriously perfect delay would be worse.

  // ---- Factor 4: Protocol deviation flag (Definition 4) ----
  // Section 5.1.D: "If Count_rcheat > delta, s7 = 1, otherwise s7 = 0", and
  // s8 = REVS(s7). Always evaluated on the current period's count, because
  // delta is described as absorbing false detections within one period.
  const bool deviates = (deviationCount > m_cheatThreshold);
  S[P7_ROUTE_BAD] = deviates ? 1.0 : 0.0;
  S[P8_ROUTE_OK]  = deviates ? 0.0 : 1.0;

  return S;
}

// ============================================================================
// Fuzzy Petri net core -- Algorithm 1
// ============================================================================

EvaluationVector
IolsrDefenseFpnt::RunFuzzyPetriNet (const std::vector<double>& s0) const
{
  NS_ASSERT (s0.size () == static_cast<size_t> (NUM_PLACES));

  std::vector<double> S = s0;

  for (int k = 0; k < MAX_FPN_ITERATIONS; ++k)
    {
      // Step 1: I = W^T * S, the equivalent input of each transition. The
      // weighted SUM, per the Type-1 rule definition sq = mu * (s1*w1 + ...).
      std::vector<double> I (NUM_TRANSITIONS, 0.0);
      for (int r = 0; r < NUM_TRANSITIONS; ++r)
        {
          for (int p = 0; p < NUM_PLACES; ++p)
            {
              I[r] += W_T[r][p] * S[p];
            }
        }

      // Step 2: G = I (x) TH   (Definition 5).
      const std::vector<double> G = MatrixOp_Threshold (I, TH);

      // Step 3: S_calc = U (o) G   (Definition 7).
      const std::vector<double> S_calc = MatrixOp_WeightedMax (U_MAT, G);

      // Step 4: S_next = S (.) S_calc   (Definition 6).
      std::vector<double> S_next = MatrixOp_Max (S, S_calc);

      // Step 5: convergence check.
      if (S_next == S)
        {
          break;
        }
      S = std::move (S_next);
    }

  // Step 6: extract the evaluation triple.
  const double sTrust    = S[P15_TRUST];
  const double sDistrust = S[P14_DISTRUST];

  double eTrust, eDistrust, eUncertain;
  if (sTrust + sDistrust < 1.0)
    {
      eTrust     = sTrust;
      eDistrust  = sDistrust;
      eUncertain = 1.0 - sTrust - sDistrust;
    }
  else
    {
      eTrust     = sTrust;
      eDistrust  = 1.0 - sTrust;
      eUncertain = 0.0;
    }

  EvaluationVector ev;
  ev.trust     = static_cast<uint8_t> (std::round (eTrust     * 255.0));
  ev.distrust  = static_cast<uint8_t> (std::round (eDistrust  * 255.0));
  ev.uncertain = static_cast<uint8_t> (std::round (eUncertain * 255.0));
  return ev;
}

// ============================================================================
// Slander filter -- Equations (1)-(3)
// ============================================================================

void
IolsrDefenseFpnt::AggregateEvaluations (const std::vector<EvaluationVector>& evs,
                                       double& outTrust,
                                       double& outUncertain) const
{
  const size_t n = evs.size ();
  NS_ASSERT_MSG (n > 0, "AggregateEvaluations requires at least one vector");

  std::vector<double> t (n), d (n), u (n);
  for (size_t k = 0; k < n; ++k)
    {
      t[k] = evs[k].trust     / 255.0;
      d[k] = evs[k].distrust  / 255.0;
      u[k] = evs[k].uncertain / 255.0;
    }

  // Equation (1): DIF_k = sum over type in {trust, distrust, uncertain} and
  // over every index v of |E^type_k - E^type_v| (v = k contributes 0).
  std::vector<double> dif (n, 0.0);
  for (size_t k = 0; k < n; ++k)
    {
      for (size_t v = 0; v < n; ++v)
        {
          dif[k] += std::abs (t[k] - t[v])
                  + std::abs (d[k] - d[v])
                  + std::abs (u[k] - u[v]);
        }
    }

  // In this pairwise form dif[k] == 0 for one k implies every recommendation
  // is identical, so dif is uniformly 0 and Equation (2) would divide by zero.
  const double maxDif = *std::max_element (dif.begin (), dif.end ());

  std::vector<double> alpha (n);
  if (maxDif < DIF_EPS)
    {
      std::fill (alpha.begin (), alpha.end (), 1.0 / static_cast<double> (n));
    }
  else
    {
      // Equation (2): alpha_k = (1/dif_k) / sum_i (1/dif_i).
      double sumInvDif = 0.0;
      for (size_t k = 0; k < n; ++k)
        {
          sumInvDif += 1.0 / dif[k];
        }
      for (size_t k = 0; k < n; ++k)
        {
          alpha[k] = (1.0 / dif[k]) / sumInvDif;
        }
    }

  // Equation (3). The distrust component is not propagated because Equation
  // (4) consumes only E_trust and E_uncertain.
  outTrust     = 0.0;
  outUncertain = 0.0;
  for (size_t k = 0; k < n; ++k)
    {
      outTrust     += alpha[k] * t[k];
      outUncertain += alpha[k] * u[k];
    }

  outTrust     = std::clamp (outTrust,     0.0, 1.0);
  outUncertain = std::clamp (outUncertain, 0.0, 1.0);
}

// ============================================================================
// Matrix operators -- Definitions 5, 6, 7
// ============================================================================

std::vector<double>
IolsrDefenseFpnt::MatrixOp_Threshold (const std::vector<double>& input,
                                     const std::vector<double>& threshold) const
{
  // Definition 5: c_i = a_i if a_i > b_i, otherwise 0.
  NS_ASSERT (input.size () == threshold.size ());
  std::vector<double> result (input.size (), 0.0);
  for (size_t i = 0; i < input.size (); ++i)
    {
      result[i] = (input[i] > threshold[i]) ? input[i] : 0.0;
    }
  return result;
}

std::vector<double>
IolsrDefenseFpnt::MatrixOp_Max (const std::vector<double>& a,
                               const std::vector<double>& b) const
{
  // Definition 6: c_i = max(a_i, b_i).
  NS_ASSERT (a.size () == b.size ());
  std::vector<double> result (a.size ());
  for (size_t i = 0; i < a.size (); ++i)
    {
      result[i] = std::max (a[i], b[i]);
    }
  return result;
}

std::vector<double>
IolsrDefenseFpnt::MatrixOp_WeightedMax (const std::vector<std::vector<double>>& Umat,
                                       const std::vector<double>& G) const
{
  // Definition 7: c_p = max over r of U[p][r] * G[r].
  std::vector<double> result (NUM_PLACES, 0.0);
  for (int p = 0; p < NUM_PLACES; ++p)
    {
      double maxVal = 0.0;
      for (int r = 0; r < NUM_TRANSITIONS; ++r)
        {
          maxVal = std::max (maxVal, Umat[p][r] * G[r]);
        }
      result[p] = maxVal;
    }
  return result;
}

// ============================================================================
// MAC -> IPv4 resolution for promiscuous monitoring
//
// Two tiers:
//   1) ARP cache -- fast, but only holds IPs we have sent unicast to.
//   2) NodeList scan anchored by our OLSR neighbor / 2-hop sets.
//
// The second tier exists because trust monitoring is a passive listen. The
// MPRs of node B never send unicast to B (B chose them, not the other way
// round), so their ARP caches never hold an entry for B. Without the
// fallback every promiscuous observation of B by its own MPRs is discarded,
// m_metrics[B] stays empty, no direct evaluation is ever produced, and
// GetNodeTrust(B) returns beta forever -- the routing layer then behaves as
// if no trust evidence existed at all.
//
// Note that tier 2 reads simulator-global state that a real node could not
// see. It is a simulation convenience for MAC-to-IP resolution only: no
// behavioral evidence is taken from it, and the set of addresses it will
// return is restricted to what our own OLSR state already knows about.
// ============================================================================

Ipv4Address
IolsrDefenseFpnt::MacToIpv4 (Mac48Address mac)
{
  if (!m_protocol)
    {
      return Ipv4Address::GetAny ();
    }

  Ptr<Node> myNode = m_protocol->GetObject<Node> ();
  if (!myNode)
    {
      return Ipv4Address::GetAny ();
    }

  // ---------- Tier 1: ARP cache.
  Ptr<Ipv4L3Protocol> l3 = myNode->GetObject<Ipv4L3Protocol> ();
  if (l3)
    {
      const NeighborSet& neighbors = m_protocol->GetNeighbors ();
      for (uint32_t i = 0; i < l3->GetNInterfaces (); ++i)
        {
          Ptr<Ipv4Interface> interface = l3->GetInterface (i);
          Ptr<ArpCache> arp = interface->GetArpCache ();
          if (!arp)
            {
              continue;
            }

          for (const auto& nb : neighbors)
            {
              ArpCache::Entry* entry = arp->Lookup (nb.neighborMainAddr);
              if (entry && entry->IsAlive () && entry->GetMacAddress () == mac)
                {
                  return nb.neighborMainAddr;
                }
            }
        }
    }

  // ---------- Tier 2: NodeList scan, anchored by OLSR state.
  std::set<Ipv4Address> recognised;
  for (const auto& nb : m_protocol->GetNeighbors ())
    {
      recognised.insert (nb.neighborMainAddr);
    }
  for (const auto& nb2 : m_protocol->GetTwoHopNeighbors ())
    {
      recognised.insert (nb2.neighborMainAddr);
      recognised.insert (nb2.twoHopNeighborAddr);
    }

  for (auto it = NodeList::Begin (); it != NodeList::End (); ++it)
    {
      Ptr<Node> other = *it;
      if (other == myNode)
        {
          continue;
        }

      bool macHere = false;
      for (uint32_t d = 0; d < other->GetNDevices (); ++d)
        {
          Ptr<WifiNetDevice> wifi = DynamicCast<WifiNetDevice> (other->GetDevice (d));
          if (wifi && Mac48Address::ConvertFrom (wifi->GetAddress ()) == mac)
            {
              macHere = true;
              break;
            }
        }
      if (!macHere)
        {
          continue;
        }

      Ptr<Ipv4> ipv4 = other->GetObject<Ipv4> ();
      if (!ipv4)
        {
          continue;
        }
      for (uint32_t i = 1; i < ipv4->GetNInterfaces (); ++i)
        {
          for (uint32_t a = 0; a < ipv4->GetNAddresses (i); ++a)
            {
              Ipv4Address addr = ipv4->GetAddress (i, a).GetLocal ();
              if (recognised.count (addr))
                {
                  return addr;
                }
            }
        }
      // MAC matched a node our OLSR state does not know about: report
      // nothing rather than inventing evidence about a stranger.
      return Ipv4Address::GetAny ();
    }

  return Ipv4Address::GetAny ();
}

} // namespace iolsr
} // namespace ns3
