#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
arm_spec.py -- the observable space of the single-vantage listener.

The pipeline (defense_detection_v2.py) addresses its inputs by a fixed list of
canonical metric names. The listener arm fills those names with what one
passive vantage point computes from the frames its own radio receives: each
canonical name below maps to an expression over the columns of the simulator's
observer_metrics-<id>.csv, evaluated on the one observer row chosen for the run
(make_arm_bundles.py). Canonical names with no genuine single-vantage analogue
are absent from the space, not proxied: the end-to-end delivery quantities
(PacketDeliveryRatio, PacketLossRatio, RxTxPacketRatio, AvgFlowLossRate,
FlowLossRateStd), the end-to-end delay and jitter quantities
(AverageEndToEndDelay, AverageJitter, AvgFlowDelay, AvgFlowJitter, FlowDelayStd,
FlowJitterStd), and FlowCount, whose only listener analogue (sources heard) is
range-dependent.

Several canonical names are borrowed from network-wide metrics of an earlier
version of this study; paper_names.py maps each column to the name the paper
uses, and the appendix feature table defines every one of them.

CANONICAL_33 is the pipeline's name order, kept so that the listener columns
are emitted in a stable order (LISTENER_ORDER).

An expression is 'obs:<expr>' with column names, numeric literals and the
operators * and /.
"""

# The pipeline's canonical names, in its order. Only the ones in LISTENER are
# ever emitted.
CANONICAL_33 = [
    "TcMessageRate", "MidMessageRate", "HnaMessageRate",
    "AverageAdvertisedLinksPerTCMessage",
    "NormalizedRoutingLoad", "RoutingOverheadRatio", "RoutingOverheadBytesRatio",
    "PacketDeliveryRatio", "PacketLossRatio", "AverageEndToEndDelay", "AverageJitter",
    "Throughput", "AverageHopCount", "DataPacketRate", "RxTxPacketRatio",
    "FlowCount", "AvgFlowDuration", "FlowDurationStd", "AvgFlowThroughput",
    "AvgFlowDelay", "AvgFlowJitter", "AvgFlowLossRate", "FlowThroughputStd",
    "FlowDelayStd", "FlowJitterStd", "FlowLossRateStd",
    "AvgTxBytesPerFlow", "AvgRxBytesPerFlow", "AvgTxPacketsPerFlow", "AvgRxPacketsPerFlow",
    "AvgTxPacketSize", "AvgRxPacketSize", "AverageMprCount",
]

# ---- THE LISTENER (21 canonical names, 17 distinct observables) --------------
# Canonical name -> observer expression over observer_metrics-<id>.csv columns.
# The four Tx/Rx pairs and the two overhead names carry the same expression: a
# listener hears each frame once on the medium and cannot separate transmitted
# from received copies; the pipeline's correlation filter removes the duplicates.
LISTENER = {
    "TcMessageRate":                      "obs:ObsTcGenerationRate",
    "MidMessageRate":                     "obs:MidCount/Duration",
    "HnaMessageRate":                     "obs:HnaCount/Duration",
    "AverageAdvertisedLinksPerTCMessage": "obs:AvgAdvertisedLinksPerTC",
    "NormalizedRoutingLoad":              "obs:ObsOlsrFrames/ObsNonOlsrDataFrames",
    "RoutingOverheadRatio":               "obs:ObsControlBytesRatio",
    "RoutingOverheadBytesRatio":          "obs:ObsControlBytesRatio",
    "Throughput":                         "obs:ObsDataBytes*8/Duration",
    "AverageHopCount":                    "obs:ObsAvgTcHopCount",
    "DataPacketRate":                     "obs:SniffedFrameRate",
    "AvgFlowDuration":                    "obs:ObsSrcDurationMean",
    "FlowDurationStd":                    "obs:ObsSrcDurationStd",
    "AvgFlowThroughput":                  "obs:ObsAvgBytesPerSource*8/Duration",
    "FlowThroughputStd":                  "obs:SrcRateStd",
    "AvgTxBytesPerFlow":                  "obs:ObsAvgBytesPerSource",
    "AvgRxBytesPerFlow":                  "obs:ObsAvgBytesPerSource",
    "AvgTxPacketsPerFlow":                "obs:ObsAvgFramesPerSource",
    "AvgRxPacketsPerFlow":                "obs:ObsAvgFramesPerSource",
    "AvgTxPacketSize":                    "obs:SniffedBytes/SniffedFrames",
    "AvgRxPacketSize":                    "obs:SniffedBytes/SniffedFrames",
    # Advertised links this listener heard over the number of distinct node
    # addresses it perceived, so injected fictitious nodes enter both terms
    # (the paper's AdvertisedLinksPerKnownNode).
    "AverageMprCount":                    "obs:SumAdvertisedLinks/DistinctAddrsSeen",
}

# Order the listener columns by their canonical rank, for stable feature matrices.
LISTENER_ORDER = [m for m in CANONICAL_33 if m in LISTENER]

ARMS = {
    "listener": {"space": LISTENER, "order": LISTENER_ORDER, "kind": "observer"},
}


if __name__ == "__main__":
    for name, a in ARMS.items():
        print(f"{name:10s}: {len(a['order'])} canonical names, "
              f"{len(set(a['space'].values()))} distinct expressions")
