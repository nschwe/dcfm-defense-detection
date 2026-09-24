#ifndef IOLSR_DEFENSE_STRATEGY_H
#define IOLSR_DEFENSE_STRATEGY_H

#include "ns3/object.h"
#include "ns3/packet.h"
#include "ns3/ipv4-address.h"
#include "ns3/mac48-address.h"
#include "ns3/ipv4-header.h"
#include "iolsr-header.h"
#include <set>
#include <vector>

namespace ns3 {
namespace iolsr {

class RoutingProtocol;

enum DropReason : uint8_t {
    DROP_NO_ROUTE = 0,
    DROP_TTL_EXPIRED = 1,
    DROP_QUEUE_FULL = 2
};

class IolsrDefenseStrategy : public Object
{
public:
  static TypeId GetTypeId(void);
  virtual ~IolsrDefenseStrategy() {}

  virtual void Setup(RoutingProtocol* proto, Ipv4Address nodeAddress) = 0;
  virtual void DoDispose() = 0;

  virtual bool IsMalicious(Ipv4Address addr) = 0;
  virtual std::set<Ipv4Address> GetBlacklist() const = 0;

  // --- Trust-routing hooks (FPNT-OLSR, Tan et al. 2015) ---------------------
  // Non-pure, so a defense that does not participate in trust routing (GCOP,
  // the null strategy) needs no changes at all.

  /// @brief Evaluation vectors to piggyback onto an outgoing TC, one per
  ///        advertised neighbor. An empty return leaves the TC in plain
  ///        RFC 3626 form.
  virtual std::vector<EvaluationVector> GetEvaluationVectors(
      const std::vector<Ipv4Address>& neighbors)
  {
    return {};
  }

  /// @brief Evaluation vectors extracted from a received TC (recommendations).
  virtual void OnRecvEvaluationVectors(Ipv4Address sender,
                                       const std::vector<Ipv4Address>& advertisedNeighbors,
                                       const std::vector<EvaluationVector>& vectors)
  {
  }

  /// @brief Trust value T(V_j) in [0,1], the node weight of the max-path-trust
  ///        routing algorithm. 1.0 = fully trusted.
  virtual double GetNodeTrust(Ipv4Address node)
  {
    return 1.0;
  }

  /// @brief Whether RoutingTableComputation should be replaced by the trust
  ///        based routing algorithm. False keeps stock RFC 3626 routing.
  virtual bool IsTrustRoutingEnabled() const
  {
    return false;
  }

  // --- Control Plane Hooks ---
  virtual void OnRecvHello(Ipv4Address senderAddress,
                           Ptr<const Packet> packet, 
                           const MessageHeader& msg, 
                           const MessageHeader::Hello& hello) = 0;

  virtual void OnRecvTc (Ipv4Address senderIfaceAddr, 
                         Ptr<const Packet> packet, 
                         const MessageHeader& msg, 
                         const MessageHeader::Tc& tc) = 0;

  virtual void OnTcGenerated(const MessageHeader::Tc& tc) = 0;

  // --- Data Plane Hooks ---
  virtual void OnDataPacketReceived(Ptr<const Packet> packet,
                                     Ipv4Address source,
                                     Ipv4Address destination,
                                     Ipv4Address nextHop) = 0;

  virtual void OnDataPacketForwarded(Ptr<const Packet> packet,
                                      Ipv4Address nextHop,
                                      Ipv4Address finalDest) = 0;

  /**
   * @brief Header-carrying variant, called from RouteInput/RouteOutput.
   *
   * RouteInput hands the routing layer a packet whose IPv4 header has already
   * been stripped, so a defense that has to fingerprint the datagram --
   * FPNT-OLSR matches an arrival against the neighbor's later retransmission
   * to measure forwarding delay -- cannot recover the header on its own.
   * Defenses that do not need it inherit this default, which drops the header
   * and calls the three-argument form. A subclass overriding either form must
   * pull both into scope with
   * `using IolsrDefenseStrategy::OnDataPacketForwarded;`.
   */
  virtual void OnDataPacketForwarded(const Ipv4Header& header,
                                     Ptr<const Packet> packet,
                                     Ipv4Address nextHop,
                                     Ipv4Address finalDest)
  {
    OnDataPacketForwarded(packet, nextHop, finalDest);
  }

  virtual void OnDataPacketDropped(Ptr<const Packet> packet, 
                                    Ipv4Address source,
                                    Ipv4Address destination,
                                    DropReason reason) = 0;

  // --- Sniffer / Promiscuous Hooks ---
  virtual void OnNeighborForwardedPacket(Mac48Address transmitter,
                                         Mac48Address receiver, Ptr<const Packet> packet) = 0;

  // --- Cross Layer & Physical Metrics ---
  virtual void OnQueueStatusReport(uint32_t size, uint32_t capacity) = 0;
  virtual void OnEnergyStateUpdate(double remainingEnergyJoules, double energyFraction) = 0;
  virtual void OnMacTxFailure(Ipv4Address neighbor, uint32_t count) = 0;

  // --- NEW: Cooperative Detection Extensions (Cross-Layer) ---
  // Reports local physical layer drops (noise/interference) to assess self-reliability
  virtual void OnSelfReliabilityReport(uint32_t localDropsCount) = 0;
  
  // Reports RTS frames seen by the sniffer (Algorithm 1)
  virtual void OnRtsReceived(Mac48Address sender, Mac48Address receiver) = 0;
  
  // Reports CTS frames seen by the sniffer (Algorithm 1)
  virtual void OnCtsReceived(Mac48Address receiver) = 0;

  virtual void PeriodicCheck() = 0;

  // Determines whether the current topology requires injecting a fictitious node
  // Returns true if a fictitious node should be added to HELLO/TC messages
  virtual bool RequiresFictitiousNode() = 0;
};

// --- Null Implementation (Default) ---
class IolsrDefenseNull : public IolsrDefenseStrategy
{
public:
  static TypeId GetTypeId(void);

  virtual void Setup(RoutingProtocol* proto, Ipv4Address nodeAddress) override {}
  virtual void DoDispose() override {}
  virtual bool IsMalicious(Ipv4Address addr) override { return false; }
  virtual std::set<Ipv4Address> GetBlacklist() const override { return {}; }

  virtual void OnRecvHello(Ipv4Address, Ptr<const Packet>, const MessageHeader&, 
                           const MessageHeader::Hello&) override {}
  virtual void OnRecvTc(Ipv4Address, Ptr<const Packet>, 
                        const MessageHeader&, const MessageHeader::Tc&) override {}
  virtual void OnTcGenerated(const MessageHeader::Tc&) override {}

  virtual void OnDataPacketReceived(Ptr<const Packet>, Ipv4Address, Ipv4Address, 
                                     Ipv4Address) override {}
  using IolsrDefenseStrategy::OnDataPacketForwarded;
  virtual void OnDataPacketForwarded(Ptr<const Packet>, Ipv4Address, Ipv4Address) override {}
  
  virtual void OnDataPacketDropped(Ptr<const Packet>, Ipv4Address, Ipv4Address, DropReason) override {}

  virtual void OnNeighborForwardedPacket(Mac48Address, Mac48Address, Ptr<const Packet>) override {}
  virtual void OnQueueStatusReport(uint32_t, uint32_t) override {}
  virtual void OnEnergyStateUpdate(double, double) override {}
  virtual void OnMacTxFailure(Ipv4Address, uint32_t) override {}
  
  // New empty implementations for the null strategy
  virtual void OnSelfReliabilityReport(uint32_t) override {}
  virtual void OnRtsReceived(Mac48Address, Mac48Address) override {}
  virtual void OnCtsReceived(Mac48Address) override {}

  virtual void PeriodicCheck() override {}

  virtual bool RequiresFictitiousNode() override { return false; }
};

} 
} 

#endif