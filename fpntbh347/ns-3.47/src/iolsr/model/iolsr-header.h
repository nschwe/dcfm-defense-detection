/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007 INESC Porto
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Author: Gustavo J. A. M. Carneiro  <gjc@inescporto.pt>
 */

#ifndef IOLSR_HEADER_H
#define IOLSR_HEADER_H

#include <stdint.h>
#include <vector>
#include "ns3/header.h"
#include "ns3/ipv4-address.h"
#include "ns3/nstime.h"


namespace ns3 {
namespace iolsr {

/**
 * @brief Evaluation Vector for FPNT-OLSR trust propagation (Tan et al. 2015).
 *
 * Ported verbatim from the audited fpnt347 `olsr` implementation. Carries the
 * (E_trust, E_distrust, E_uncertain) triple produced by Algorithm 1, quantized
 * to uint8_t so piggybacking it onto a TC costs 4 bytes per advertised
 * neighbour (paper Fig. 6).
 *
 * NOTE for the listener: this extends the TC wire format. The passive observer
 * parses this same MessageHeader, so an FPNT-enabled TC stays parseable, and
 * the extra bytes are exactly the control-plane footprint the detector may
 * pick up. That is the mechanism under test, not a side effect.
 */
struct EvaluationVector
{
  uint8_t trust;      ///< Quantized E_trust     (truth degree of p15)
  uint8_t distrust;   ///< Quantized E_distrust  (truth degree of p14)
  uint8_t uncertain;  ///< Quantized E_uncertain (1 - p14 - p15)
  uint8_t reserved;   ///< Padding for 32-bit alignment

  EvaluationVector ()
    : trust (0), distrust (0), uncertain (0), reserved (0)
  {
  }
};

double EmfToSeconds (uint8_t emf);
uint8_t SecondsToEmf (double seconds);

// 3.3.  Packet Format
//
//    The basic layout of any packet in IOLSR is as follows (omitting IP and
//    UDP headers):
//
//        0                   1                   2                   3
//        0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
//       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//       |         Packet Length         |    Packet Sequence Number     |
//       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//       |  Message Type |     Vtime     |         Message Size          |
//       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//       |                      Originator Address                       |
//       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//       |  Time To Live |   Hop Count   |    Message Sequence Number    |
//       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//       |                                                               |
//       :                            MESSAGE                            :
//       |                                                               |
//       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//       |  Message Type |     Vtime     |         Message Size          |
//       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//       |                      Originator Address                       |
//       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//       |  Time To Live |   Hop Count   |    Message Sequence Number    |
//       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//       |                                                               |
//       :                            MESSAGE                            :
//       |                                                               |
//       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//       :                                                               :
//                (etc.)
class PacketHeader : public Header
{
public:
  PacketHeader ();
  virtual ~PacketHeader ();

  void SetPacketLength (uint16_t length)
  {
    m_packetLength = length;
  }
  uint16_t GetPacketLength () const
  {
    return m_packetLength;
  }

  void SetPacketSequenceNumber (uint16_t seqnum)
  {
    m_packetSequenceNumber = seqnum;
  }
  uint16_t GetPacketSequenceNumber () const
  {
    return m_packetSequenceNumber;
  }

private:
  uint16_t m_packetLength;
  uint16_t m_packetSequenceNumber;

public:
  static TypeId GetTypeId (void);
  virtual TypeId GetInstanceTypeId (void) const;
  virtual void Print (std::ostream &os) const;
  virtual uint32_t GetSerializedSize (void) const;
  virtual void Serialize (Buffer::Iterator start) const;
  virtual uint32_t Deserialize (Buffer::Iterator start);
};


//        0                   1                   2                   3
//        0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
//       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//       |  Message Type |     Vtime     |         Message Size          |
//       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//       |                      Originator Address                       |
//       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//       |  Time To Live |   Hop Count   |    Message Sequence Number    |
//       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
class MessageHeader : public Header
{
public:

  enum MessageType {
    HELLO_MESSAGE = 1,
    TC_MESSAGE    = 2,
    MID_MESSAGE   = 3,
    HNA_MESSAGE   = 4,
  };

  MessageHeader ();
  virtual ~MessageHeader ();

  void SetMessageType (MessageType messageType)
  {
    m_messageType = messageType;
  }
  MessageType GetMessageType () const
  {
    return m_messageType;
  }

  void SetVTime (Time time)
  {
    m_vTime = SecondsToEmf (time.GetSeconds ());
  }
  Time GetVTime () const
  {
    return Seconds (EmfToSeconds (m_vTime));
  }

  void SetOriginatorAddress (Ipv4Address originatorAddress)
  {
    m_originatorAddress = originatorAddress;
  }
  Ipv4Address GetOriginatorAddress () const
  {
    return m_originatorAddress;
  }

  void SetTimeToLive (uint8_t timeToLive)
  {
    m_timeToLive = timeToLive;
  }
  uint8_t GetTimeToLive () const
  {
    return m_timeToLive;
  }

  void SetHopCount (uint8_t hopCount)
  {
    m_hopCount = hopCount;
  }
  uint8_t GetHopCount () const
  {
    return m_hopCount;
  }

  void SetMessageSequenceNumber (uint16_t messageSequenceNumber)
  {
    m_messageSequenceNumber = messageSequenceNumber;
  }
  uint16_t GetMessageSequenceNumber () const
  {
    return m_messageSequenceNumber;
  }

//   void SetMessageSize (uint16_t messageSize)
//   {
//     m_messageSize = messageSize;
//   }
//   uint16_t GetMessageSize () const
//   {
//     return m_messageSize;
//   }

private:
  MessageType m_messageType;
  uint8_t m_vTime;
  Ipv4Address m_originatorAddress;
  uint8_t m_timeToLive;
  uint8_t m_hopCount;
  uint16_t m_messageSequenceNumber;
  uint16_t m_messageSize;

public:
  static TypeId GetTypeId (void);
  virtual TypeId GetInstanceTypeId (void) const;
  virtual void Print (std::ostream &os) const;
  virtual uint32_t GetSerializedSize (void) const;
  virtual void Serialize (Buffer::Iterator start) const;
  virtual uint32_t Deserialize (Buffer::Iterator start);

  // 5.1.  MID Message Format
  //
  //    The proposed format of a MID message is as follows:
  //
  //        0                   1                   2                   3
  //        0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
  //       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  //       |                    IOLSR Interface Address                     |
  //       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  //       |                    IOLSR Interface Address                     |
  //       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  //       |                              ...                              |
  //       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  struct Mid
  {
    std::vector<Ipv4Address> interfaceAddresses;
    void Print (std::ostream &os) const;
    uint32_t GetSerializedSize (void) const;
    void Serialize (Buffer::Iterator start) const;
    uint32_t Deserialize (Buffer::Iterator start, uint32_t messageSize);
  };

  // 6.1.  HELLO Message Format
  //
  //        0                   1                   2                   3
  //        0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
  //
  //       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  //       |          Reserved             |     Htime     |  Willingness  |
  //       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  //       |   Link Code   |   Reserved    |       Link Message Size       |
  //       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  //       |                  Neighbor Interface Address                   |
  //       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  //       |                  Neighbor Interface Address                   |
  //       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  //       :                             .  .  .                           :
  //       :                                                               :
  //       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  //       |   Link Code   |   Reserved    |       Link Message Size       |
  //       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  //       |                  Neighbor Interface Address                   |
  //       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  //       |                  Neighbor Interface Address                   |
  //       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  //       :                                                               :
  //       :                                       :
  //    (etc.)
  struct Hello
  {
    struct LinkMessage {
      uint8_t linkCode;
      std::vector<Ipv4Address> neighborInterfaceAddresses;
    };

    uint8_t hTime;
    void SetHTime (Time time)
    {
      this->hTime = SecondsToEmf (time.GetSeconds ());
    }
    Time GetHTime () const
    {
      return Seconds (EmfToSeconds (this->hTime));
    }

    uint8_t willingness;
    std::vector<LinkMessage> linkMessages;

    void Print (std::ostream &os) const;
    uint32_t GetSerializedSize (void) const;
    void Serialize (Buffer::Iterator start) const;
    uint32_t Deserialize (Buffer::Iterator start, uint32_t messageSize);
  };

  // 9.1.  TC Message Format
  //
  //    The proposed format of a TC message is as follows:
  //
  //        0                   1                   2                   3
  //        0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
  //       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  //       |              ANSN             |           Reserved            |
  //       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  //       |               Advertised Neighbor Main Address                |
  //       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  //       |               Advertised Neighbor Main Address                |
  //       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  //       |                              ...                              |
  //       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

  struct Tc
  {
    std::vector<Ipv4Address> neighborAddresses;

    /// FPNT-OLSR extension (paper Section 5.2 / Fig. 6): one evaluation vector
    /// per entry of neighborAddresses. Serialized only when the two vectors
    /// have equal length; otherwise the TC goes out in plain RFC 3626 form.
    std::vector<EvaluationVector> evaluationVectors;

    /// Whether this TC actually carries FPNT evaluation vectors on the wire.
    /// Serialize() and GetSerializedSize() must agree exactly or ns-3 writes
    /// past the buffer it reserved; both consult this predicate.
    bool CarriesEvaluationVectors () const
    {
      return !evaluationVectors.empty () &&
             evaluationVectors.size () == neighborAddresses.size ();
    }
    uint16_t ansn;

    void Print (std::ostream &os) const;
    uint32_t GetSerializedSize (void) const;
    void Serialize (Buffer::Iterator start) const;
    uint32_t Deserialize (Buffer::Iterator start, uint32_t messageSize);
  };


  // 12.1.  HNA Message Format
  //
  //    The proposed format of an HNA-message is:
  //
  //        0                   1                   2                   3
  //        0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
  //       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  //       |                         Network Address                       |
  //       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  //       |                             Netmask                           |
  //       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  //       |                         Network Address                       |
  //       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  //       |                             Netmask                           |
  //       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
  //       |                              ...                              |
  //       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

  // Note: HNA stands for Host Network Association
  struct Hna
  {
    struct Association
    {
      Ipv4Address address;
      Ipv4Mask mask;
    };
    std::vector<Association> associations;

    void Print (std::ostream &os) const;
    uint32_t GetSerializedSize (void) const;
    void Serialize (Buffer::Iterator start) const;
    uint32_t Deserialize (Buffer::Iterator start, uint32_t messageSize);
  };

private:
  struct
  {
    Mid mid;
    Hello hello;
    Tc tc;
    Hna hna;
  } m_message; // union not allowed

public:

  Mid& GetMid ()
  {
    if (m_messageType == 0)
      {
        m_messageType = MID_MESSAGE;
      }
    else
      {
        NS_ASSERT (m_messageType == MID_MESSAGE);
      }
    return m_message.mid;
  }

  Hello& GetHello ()
  {
    if (m_messageType == 0)
      {
        m_messageType = HELLO_MESSAGE;
      }
    else
      {
        NS_ASSERT (m_messageType == HELLO_MESSAGE);
      }
    return m_message.hello;
  }

  Tc& GetTc ()
  {
    if (m_messageType == 0)
      {
        m_messageType = TC_MESSAGE;
      }
    else
      {
        NS_ASSERT (m_messageType == TC_MESSAGE);
      }
    return m_message.tc;
  }

  Hna& GetHna ()
  {
    if (m_messageType == 0)
      {
        m_messageType = HNA_MESSAGE;
      }
    else
      {
        NS_ASSERT (m_messageType == HNA_MESSAGE);
      }
    return m_message.hna;
  }


  const Mid& GetMid () const
  {
    NS_ASSERT (m_messageType == MID_MESSAGE);
    return m_message.mid;
  }

  const Hello& GetHello () const
  {
    NS_ASSERT (m_messageType == HELLO_MESSAGE);
    return m_message.hello;
  }

  const Tc& GetTc () const
  {
    NS_ASSERT (m_messageType == TC_MESSAGE);
    return m_message.tc;
  }

  const Hna& GetHna () const
  {
    NS_ASSERT (m_messageType == HNA_MESSAGE);
    return m_message.hna;
  }


};


static inline std::ostream& operator<< (std::ostream& os, const PacketHeader & packet)
{
  packet.Print (os);
  return os;
}

static inline std::ostream& operator<< (std::ostream& os, const MessageHeader & message)
{
  message.Print (os);
  return os;
}

typedef std::vector<MessageHeader> MessageList;

static inline std::ostream& operator<< (std::ostream& os, const MessageList & messages)
{
  os << "[";
  for (std::vector<MessageHeader>::const_iterator messageIter = messages.begin ();
       messageIter != messages.end (); messageIter++)
    {
      messageIter->Print (os);
      if (messageIter+1 != messages.end ())
        os << ", ";
    }
  os << "]";
  return os;
}


}
}  // namespace iolsr, ns3

#endif /* IOLSR_HEADER_H */

