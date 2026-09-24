/*
 * Copyright (c) 2024 NS-3 Security Extension Project
 *
 * Author: Oded Ofek <odedofek2@gmail.com>
 */

#include "iolsr-defense-strategy.h"
#include "ns3/log.h"

namespace ns3 {
namespace iolsr {

NS_LOG_COMPONENT_DEFINE("IolsrDefenseStrategy");

NS_OBJECT_ENSURE_REGISTERED(IolsrDefenseStrategy);
NS_OBJECT_ENSURE_REGISTERED(IolsrDefenseNull);

TypeId
IolsrDefenseStrategy::GetTypeId(void)
{
  static TypeId tid = TypeId("ns3::iolsr::IolsrDefenseStrategy")
    .SetParent<Object>()
    .SetGroupName("Olsr");
  return tid;
}

TypeId
IolsrDefenseNull::GetTypeId(void)
{
  static TypeId tid = TypeId("ns3::iolsr::IolsrDefenseNull")
    .SetParent<IolsrDefenseStrategy>()
    .SetGroupName("Olsr")
    .AddConstructor<IolsrDefenseNull>();
  return tid;
}

} // namespace iolsr
} // namespace ns3