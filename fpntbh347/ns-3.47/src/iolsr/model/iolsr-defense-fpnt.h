/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
#ifndef IOLSR_DEFENSE_FPNT_H
#define IOLSR_DEFENSE_FPNT_H

#include "iolsr-defense-strategy.h"
#include "ns3/nstime.h"
#include "ns3/mac48-address.h"
#include <cstddef>
#include <map>
#include <vector>

namespace ns3 {
namespace iolsr {

/**
 * \brief FPNT-OLSR trust reasoning mechanism.
 *
 * Implementation of:
 *     Tan, Li, Dong (2015). "Trust based routing mechanism for securing
 *     OLSR-based MANET", Ad Hoc Networks 30, pp. 84-98.
 *
 * This class implements:
 *   - The four trust factors (Section 3.1 / Definitions 1-4, collected as
 *     described in Section 5.1):
 *       * Load                     (p1 / p2)  -- DoS victim signature
 *       * Packet forwarding rate   (p3 / p4)  -- blackhole signature
 *       * Average forwarding delay (p5 / p6)  -- jellyfish signature
 *       * Protocol deviation flag  (p7 / p8)  -- routing-plane signature
 *
 *   - The 7 fuzzy rules over 15 propositions, decomposed into the 11
 *     transitions the paper counts ("let n = 14, m = 11"): the two OR rules
 *     R1 and R6 become three competitive Type-2 transitions each, because
 *     Type-2 semantics require an independent threshold test per input place.
 *       * R1: IF p1 OR p4 OR p5      THEN p9   (1,1,1; .4,.4,.5; .9,.9,.6)
 *       * R2: IF p2 AND p5           THEN p10  (.6,.4; .5; .8)
 *       * R3: IF p7                  THEN p11  (1; .5; .9)
 *       * R4: IF p3 AND p6 AND p2    THEN p12  (.6,.3,.1; .7; .9)
 *       * R5: IF p8                  THEN p13  (1; .8; 1)
 *       * R6: IF p9 OR p10 OR p11    THEN p14  (1,1,1; .4,.4,.4; .9,.7,.9)
 *       * R7: IF p12 AND p13         THEN p15  (.5,.5; .6; .9)
 *
 *   - The matrix-based reasoning algorithm (Algorithm 1) with operators
 *     (x)/(.)/(o) of Definitions 5, 6 and 7.
 *   - The pairwise L1 slander filter (Equations 1-3).
 *   - Equation (4) trust synthesis: T = E_trust + beta * E_uncertain.
 *   - Equation (5) temporal smoothing, with an explicit first-period
 *     bootstrap (the paper leaves T_{c-1} undefined for the first period).
 *   - The trust based routing algorithm (Algorithm 2) lives in
 *     RoutingProtocol::RunTrustDijkstra and queries this class via
 *     GetNodeTrust.
 *
 * Deviations from the paper are never silent: every behavior this class adds
 * beyond the paper's text sits behind an ns-3 attribute whose default value
 * reproduces the paper. See StickyEvidence, DemoteUnverifiedNodes,
 * RollbackOnMacFailure and MonitorTcForwarding.
 *
 * Architecture note:
 *   This class is passive regarding scheduling. The RoutingProtocol invokes
 *   PeriodicCheck() once per TrustUpdateInterval ('t' in Section 5.1).
 */
class IolsrDefenseFpnt : public IolsrDefenseStrategy
{
public:
  static TypeId GetTypeId (void);

  IolsrDefenseFpnt ();
  virtual ~IolsrDefenseFpnt ();

  // ======================================================================
  // Setup & Lifecycle
  // ======================================================================
  virtual void Setup (RoutingProtocol* proto, Ipv4Address nodeAddress) override;
  virtual void DoDispose () override;

  // ======================================================================
  // Trust Query Methods (Routing Integration)
  // ======================================================================
  virtual bool IsMalicious (Ipv4Address addr) override;
  virtual std::set<Ipv4Address> GetBlacklist () const override;
  virtual std::vector<EvaluationVector> GetEvaluationVectors (
      const std::vector<Ipv4Address> &neighbors) override;
  virtual double GetNodeTrust (Ipv4Address node) override;
  virtual bool IsTrustRoutingEnabled () const override { return m_enabled; }

  /**
   * @brief Toggle the defense state with symmetric cold-start semantics.
   *
   * On EVERY state transition (disabled->enabled and enabled->disabled)
   * every piece of accumulated state is wiped, so neither phase can carry
   * residue from the other. Hooks fire regardless of m_enabled and
   * PeriodicCheck only clears the per-period containers, so without the
   * symmetric wipe an enabled-phase trust table or the longer-lived D1/D2
   * bookkeeping would leak into the subsequent phase. No-op calls (the same
   * value passed twice) are skipped.
   */
  void SetEnabled (bool enabled);
  bool GetEnabled () const;

  // ======================================================================
  // Read-only state introspection.
  //
  // Reports the current sizes of every accumulated-state container plus the
  // derived blacklist size, so an evaluation harness can verify that a
  // window-boundary cold start really emptied the defense state. Strictly
  // const and side-effect-free; plays no part in the trust algorithm.
  // ======================================================================
  struct DebugStateSizes
  {
    std::size_t metrics                 = 0;
    std::size_t trustTable              = 0;
    std::size_t directEvaluationVectors = 0;
    std::size_t lastSValues             = 0;
    std::size_t recommendations         = 0;
    std::size_t pendingArrivals         = 0;   // # neighbours with pending arrivals
    std::size_t lastTcTime              = 0;
    std::size_t mprSelectionTime        = 0;
    std::size_t blacklist               = 0;   // derived: nodes below threshold
  };
  DebugStateSizes GetDebugStateSizes () const;

  // ======================================================================
  // Incoming Message Handlers (trust propagation, Section 5.2)
  // ======================================================================
  virtual void OnRecvEvaluationVectors (
      Ipv4Address sender,
      const std::vector<Ipv4Address> &advertisedNeighbors,
      const std::vector<EvaluationVector> &vectors) override;

  virtual void OnRecvHello (Ipv4Address senderAddress,
                            Ptr<const Packet> packet,
                            const MessageHeader& msg,
                            const MessageHeader::Hello& hello) override;

  virtual void OnRecvTc (Ipv4Address senderIfaceAddr,
                         Ptr<const Packet> packet,
                         const MessageHeader& msg,
                         const MessageHeader::Tc& tc) override;

  virtual void OnTcGenerated (const MessageHeader::Tc& tc) override;

  // ======================================================================
  // Monitoring & metric collection (Section 5.1)
  //
  // Hook wiring summary:
  //   OnDataPacketForwarded      -> load, PFR denominator (Count^j_rcv) and
  //                                 arrival timestamp for the delay measure
  //   OnNeighborForwardedPacket  -> load and PFR denominator for traffic
  //                                 injected by OTHER transmitters, PFR
  //                                 numerator (Count^j_fwd) and the
  //                                 delay-departure match
  //   OnRecvTc                   -> deviation flag D2, optional TC-relay
  //                                 term of the PFR
  //   OnDataPacketReceived, OnDataPacketDropped, OnQueueStatusReport,
  //   OnEnergyStateUpdate        -> outside the paper's model
  // ======================================================================
  virtual void OnDataPacketReceived (Ptr<const Packet> packet,
                                     Ipv4Address source,
                                     Ipv4Address destination,
                                     Ipv4Address nextHop) override;

  using IolsrDefenseStrategy::OnDataPacketForwarded;
  virtual void OnDataPacketForwarded (Ptr<const Packet> packet,
                                      Ipv4Address nextHop,
                                      Ipv4Address finalDest) override;

  virtual void OnDataPacketForwarded (const Ipv4Header &header,
                                      Ptr<const Packet> packet,
                                      Ipv4Address nextHop,
                                      Ipv4Address finalDest) override;

  virtual void OnDataPacketDropped (Ptr<const Packet> packet,
                                    Ipv4Address source,
                                    Ipv4Address destination,
                                    DropReason reason) override;

  virtual void OnNeighborForwardedPacket (Mac48Address transmitter,
                                          Mac48Address receiver,
                                          Ptr<const Packet> packet) override;

  virtual void OnQueueStatusReport (uint32_t size, uint32_t capacity) override;
  virtual void OnEnergyStateUpdate (double remainingEnergyJoules,
                                    double energyFraction) override;
  virtual void OnMacTxFailure (Ipv4Address neighbor, uint32_t count) override;

  // Cross-layer hooks belonging to other defenses in this tree; the paper's
  // model uses none of them.
  virtual void OnSelfReliabilityReport (uint32_t localDropsCount) override {}
  virtual void OnRtsReceived (Mac48Address sender, Mac48Address receiver) override {}
  virtual void OnCtsReceived (Mac48Address receiver) override {}
  virtual bool RequiresFictitiousNode () override { return false; }

  /**
   * \brief Execute one full trust reasoning cycle.
   *
   * Performs, in order:
   *   1. Protocol-deviation rule D1 scan (silent MPR detection, Section 5.1.D).
   *   2. Expiration of unmatched delay-measurement arrivals and TC relay
   *      obligations.
   *   3. Fresh direct evaluations from the observed metrics, for every
   *      monitored neighbor (Algorithm 1).
   *   4. Aggregation of the direct evaluation plus received recommendations
   *      via the pairwise L1 slander filter (Equations 1-3).
   *   5. Equation (4) trust synthesis followed by Equation (5) temporal
   *      smoothing.
   *   6. Reroute notification if any verdict flipped this period.
   */
  virtual void PeriodicCheck () override;

private:
  RoutingProtocol* m_protocol;

  // ----------------------------------------------------------------------
  // Per-neighbor behavioral counters -- all four factors (Section 5.1).
  // ----------------------------------------------------------------------
  struct NodeBehaviorMetrics
  {
    uint32_t countLoad;   // Bytes of received traffic (Count^j_load).
    uint32_t countRcv;    // Packets V_j is expected to forward (Count^j_rcv).
    uint32_t countFwd;    // Packets observed to be forwarded by V_j (Count^j_fwd).
    uint32_t countRCheat; // Routing-plane deviations (Count^j_rcheat).
    double   totalDelay;  // Sum of forwarding delays (seconds) over matched
                          // packets; d_j of Definition 3.

    NodeBehaviorMetrics ()
      : countLoad (0), countRcv (0), countFwd (0),
        countRCheat (0), totalDelay (0.0) {}
  };

  // Per-period metric counters, one entry per monitored neighbor.
  std::map<Ipv4Address, NodeBehaviorMetrics> m_metrics;

  // Routing-plane deviation counters. Kept apart from m_metrics because they
  // are indexed by our MPRs (the nodes D1/D2 can accuse) whereas the data
  // plane counters are indexed by our MPR selectors (the nodes we monitor).
  // Merging the two lets the selector-scoped pruning in PeriodicCheck discard
  // deviation evidence before it is ever evaluated.
  std::map<Ipv4Address, uint32_t> m_deviationCounts;

  // Aggregated trust value T(V_j) for every known node in the network.
  std::map<Ipv4Address, double> m_trustTable;

  // Latest direct evaluation vectors (piggybacked onto outgoing TC messages).
  std::map<Ipv4Address, EvaluationVector> m_directEvaluationVectors;

  // Persistence store for S^(0) across periods, used only when the
  // StickyEvidence attribute is on.
  std::map<Ipv4Address, std::vector<double>> m_lastSValues;

  // Recommendations received this period, keyed by (originator, target).
  std::map<std::pair<Ipv4Address, Ipv4Address>, EvaluationVector> m_recommendations;

  // ----------------------------------------------------------------------
  // Delay-measurement bookkeeping (Definition 3).
  //
  // Observing a packet arrive at neighbor V_j records
  // arrival_time[V_j][fingerprint] = now. Observing V_j transmit a packet
  // with the same fingerprint yields delay = now - arrival_time, which is
  // added to totalDelay[V_j]. Unmatched arrivals are expired in
  // PeriodicCheck so the map cannot grow without bound.
  // ----------------------------------------------------------------------
  struct PendingArrival
  {
    Time arrivalTime;
  };
  std::map<Ipv4Address, std::map<uint64_t, PendingArrival>> m_pendingArrivals;

  // ----------------------------------------------------------------------
  // Deviation-flag (D1) bookkeeping: last-seen TC time per originator.
  // D1 fires when an MPR of this node has not emitted a TC within
  // OLSR_TOP_HOLD_TIME = 3 * tcInterval.
  // ----------------------------------------------------------------------
  std::map<Ipv4Address, Time> m_lastTcTime;

  // First time each current MPR was seen as ours, shared by D1 and D2 as the
  // start of the grace window.
  std::map<Ipv4Address, Time> m_mprSelectionTime;

  // (originator, message sequence) pairs already judged by D2. The routing
  // protocol delivers OnRecvTc for every copy of a flooded TC, so without
  // this a single omission would be counted once per relay that reaches us.
  std::set<std::pair<Ipv4Address, uint16_t>> m_seenTcForD2;

  // TC relay obligations pending observation, for the optional TC term of
  // the packet forwarding rate: (originator, message sequence) -> when it
  // was recorded, per expected relayer.
  std::map<Ipv4Address, std::map<std::pair<Ipv4Address, uint16_t>, Time>> m_pendingTcRelays;

  // Node's own main address, cached from Setup() for the D2 self-omission check.
  Ipv4Address m_selfAddress;

  // ----------------------------------------------------------------------
  // Reasoning parameters (all exposed as ns-3 attributes).
  // ----------------------------------------------------------------------
  Time   m_checkInterval;       // Trust update period 't'.
  double m_maliciousThreshold;  // Node is reported malicious iff T(V_j) < this.
  double m_uncertaintyBeta;     // Eq. (4): T = E_trust + beta * E_uncertain.
  double m_fadingFactor;        // Eq. (5): lambda in temporal smoothing.
  double m_maxLoad;             // NORM normalizer for load, bits/s (Def. 10).
  double m_maxDelay;            // NORM normalizer for avg delay, seconds.
  uint32_t m_cheatThreshold;    // delta in Section 5.1.D.
  bool   m_enabled;             // Runtime toggle; see SetEnabled.

  // ---- Opt-in behaviors that go beyond the paper. All default to the
  // ---- paper's own semantics (see GetTypeId for the rationale of each).
  bool m_stickyEvidence;        // Carry a factor's truth degree across a
                                // period in which it saw no observation.
  bool m_demoteUnverified;      // Discount nodes known only from hearsay.
  bool m_rollbackOnMacFailure;  // Undo Count_rcv when our own link failed.
  bool m_monitorTcForwarding;   // Add the TC-relay term to the PFR.
  bool m_flatTrust;             // FPNT-DIAG (temporary): every node gets beta.
  bool m_broadcastLoad;         // Definition 1: count broadcast frames too.
  bool m_retainEvaluations;     // Keep the opinion when a selector is lost.

  // ----------------------------------------------------------------------
  // Helpers (implementation in .cc).
  // ----------------------------------------------------------------------
  std::vector<double> MetricsToS0 (Ipv4Address addr,
                                   const NodeBehaviorMetrics& metrics,
                                   uint32_t deviationCount);

  EvaluationVector RunFuzzyPetriNet (const std::vector<double>& s0) const;

  /**
   * \brief Apply Equations (1)-(3): pairwise L1 DIF slander filtering.
   *
   * The direct evaluation (when present) is expected to be one element of
   * \p evs: the paper states that direct and indirect evaluations carry the
   * same importance and are aggregated identically.
   */
  void AggregateEvaluations (const std::vector<EvaluationVector>& evs,
                             double& outTrust,
                             double& outUncertain) const;

  /**
   * \brief Protocol-deviation rule D1 (Section 5.1.D, silent-MPR case).
   *
   * For every neighbor currently in this node's MPR set, check whether a TC
   * has been received within OLSR_TOP_HOLD_TIME. If not, increment that
   * MPR's routing-cheat counter.
   */
  void ScanDeviationRuleD1 ();

  /// \brief Expire unmatched delay arrivals and TC relay obligations.
  void ExpireStaleObservations ();

  /// \brief True when \p addr is one of our MPR selectors, i.e. a node the
  ///        paper puts us in charge of monitoring.
  bool IsOurMprSelector (Ipv4Address addr) const;

  /// @brief Every MPR selector of ours that our own OLSR state places on a
  ///        link with @p transmitter, i.e. inside its transmission radius,
  ///        and which therefore received the broadcast we just heard.
  ///        Section 5.1: "V_i monitors and analyzes all of V_j's traffic
  ///        within its transmission radius."
  std::set<Ipv4Address> SelectorsInRadiusOf (Ipv4Address transmitter) const;

  /// \brief OLSR_TOP_HOLD_TIME = 3 * TcInterval, read from the protocol.
  Time GetTopologyHoldTime () const;

  // Fuzzy Petri net matrix operators (Definitions 5, 6, 7).
  std::vector<double> MatrixOp_Threshold (
      const std::vector<double>& input,
      const std::vector<double>& threshold) const;

  std::vector<double> MatrixOp_Max (
      const std::vector<double>& a,
      const std::vector<double>& b) const;

  std::vector<double> MatrixOp_WeightedMax (
      const std::vector<std::vector<double>>& U,
      const std::vector<double>& G) const;

  Ipv4Address MacToIpv4 (Mac48Address mac);
};

} // namespace iolsr
} // namespace ns3

#endif /* IOLSR_DEFENSE_FPNT_H */
