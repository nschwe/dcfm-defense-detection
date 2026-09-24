#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paper_names.py -- presentation-layer naming for the listener arm.

WHY THIS FILE EXISTS
--------------------
arm_spec.py deliberately feeds every arm the SAME canonical column names, so the
learning pipeline is byte-identical across arms and any difference in the result
is attributable to the observation model alone. That is correct engineering, but
it means the listener arm reports quantities under the canonical names of the
network-wide metrics they replace, and several of those quantities are not the
same measurement.

This module maps each canonical name to the name used in the paper. It is a
LOOKUP ONLY. Nothing in the pipeline imports it, it changes no column, and it
cannot affect any result. Its purpose is that the paper's Section 3.4 table is
generated from the code rather than transcribed by hand.

NAMING RULE
-----------
A name is changed ONLY where the single-vantage quantity differs from the
network-wide metric of the same name. Names that already describe what a passive
listener computes are left untouched -- which is itself informative: it marks the
features for which nothing changed.

VERIFICATION
------------
Every entry below was checked twice:
  (a) against the accumulator (struct ObserverCounters) and the emit block in
      nv347/ns-3.47/scratch/iolsr-tests-corrected.cc
  (b) numerically, against the built bundles
      (analysis/arms_c10k_stackcal/listener/colab_data/wide_static.csv.gz,
      40,000 rows) and against a raw observer_metrics CSV, by recomputing each
      formula from the raw fields.

PENDING: this is the presentation layer only. The column names in arm_spec.py,
in CANONICAL_33 (defense_detection_v2.py:232-248), in the built bundles and in
every stored result still carry the canonical names. Renaming those is a
separate job, to be done only when no run is active. Until then this module
is a lookup: nothing in the pipeline imports it, it changes no column, and it
cannot affect any result.
"""

# canonical name -> (paper name, one-line definition, source expression)
# paper name None => the column duplicates another and is not reported.
PAPER_NAMES = {
    # ---- unchanged: the canonical name already describes what is measured ----
    "TcMessageRate": (
        "TcMessageRate",
        "Distinct TC messages generated per second, de-duplicated across flooded "
        "copies by (originator, sequence number).",
        "obs:ObsTcGenerationRate = tcUniqueCount / duration"),
    "AverageAdvertisedLinksPerTCMessage": (
        "AverageAdvertisedLinksPerTCMessage",
        "Mean number of links advertised per TC message heard.",
        "obs:AvgAdvertisedLinksPerTC = tcRows / tcCount"),
    "MidMessageRate": (
        "MidMessageRate",
        "MID messages heard per second.",
        "obs:MidCount / Duration"),
    "HnaMessageRate": (
        "HnaMessageRate",
        "HNA messages heard per second.",
        "obs:HnaCount / Duration"),
    "RoutingOverheadBytesRatio": (
        "RoutingOverheadBytesRatio",
        "OLSR control bytes divided by data bytes, over the frames heard.",
        "obs:ObsControlBytesRatio = olsrBytes / dataBytes"),

    # ---- renamed: the single-vantage quantity differs from the metric name ----
    "AverageMprCount": (
        "AdvertisedLinksPerKnownNode",
        "Sum of advertised links, over the number of distinct node addresses this "
        "vantage point has seen. NOT an MPR-selector count: under an active "
        "defence the advertised set carries the injected fictitious link, which "
        "the node's real selector set does not.",
        "obs:SumAdvertisedLinks / DistinctAddrsSeen"),
    "AverageHopCount": (
        "MeanTcHopCount",
        "Mean of the hop-count field in the headers of the TC messages received, "
        "i.e. how far in hops the TC originators heard were. NOT the mean "
        "data-path hop count.",
        "obs:ObsAvgTcHopCount = tcHopSum / tcCount"),
    "DataPacketRate": (
        "SniffedFrameRate",
        "All frames successfully received at the PHY per second, including "
        "acknowledgements and control frames.",
        "obs:SniffedFrameRate = frames / duration"),
    "Throughput": (
        "ObservedDataRate",
        "Bit rate of the IPv4-carrying frames heard at this vantage point. Not a "
        "network-wide throughput.",
        "obs:ObsDataBytes * 8 / Duration"),
    "NormalizedRoutingLoad": (
        "ControlToDataFrameRatio",
        "OLSR frames per non-OLSR data frame heard. The conventional normalised "
        "routing load divides by packets delivered, which a listener cannot "
        "observe.",
        "obs:ObsOlsrFrames / ObsNonOlsrDataFrames"),
    "RoutingOverheadRatio": (
        None,
        "Duplicate: identical expression to RoutingOverheadBytesRatio "
        "(0 differing rows in 40,000). Not reported separately.",
        "obs:ObsControlBytesRatio"),
    "AvgTxPacketSize": (
        "MeanSniffedFrameSize",
        "Mean size of a frame received at this vantage point. A listener observes "
        "each frame once on the medium and cannot separate transmitted from "
        "received copies.",
        "obs:SniffedBytes / SniffedFrames"),
    "AvgRxPacketSize": (
        None,
        "Duplicate of AvgTxPacketSize under single-vantage observation "
        "(0 differing rows in 40,000).",
        "obs:SniffedBytes / SniffedFrames"),
    "AvgFlowThroughput": (
        "MeanPerSourceDataRate",
        "Mean data rate over the sources this vantage point overheard. Per source, "
        "not per classified flow.",
        "obs:ObsAvgBytesPerSource * 8 / Duration"),
    "FlowThroughputStd": (
        "PerSourceRateStd",
        "Standard deviation of the per-source byte rates over the sources heard. "
        "The network-wide metric of the same name is computed by FlowMonitor over "
        "51 classified flows from receiver-side rxBytes.",
        "obs:SrcRateStd"),
    "AvgFlowDuration": (
        "MeanSourceVisibilityWindow",
        "Mean of (last heard - first heard) per source. A visibility window, not a "
        "flow lifetime.",
        "obs:ObsSrcDurationMean"),
    "FlowDurationStd": (
        "SourceVisibilityWindowStd",
        "Standard deviation of the per-source visibility windows.",
        "obs:ObsSrcDurationStd"),
    "AvgTxBytesPerFlow": (
        "MeanBytesPerSource",
        "Mean bytes heard per source. Per source, not per flow; Tx and Rx are not "
        "separable at one vantage point.",
        "obs:ObsAvgBytesPerSource"),
    "AvgRxBytesPerFlow": (
        None,
        "Duplicate of AvgTxBytesPerFlow (0 differing rows in 40,000).",
        "obs:ObsAvgBytesPerSource"),
    "AvgTxPacketsPerFlow": (
        "MeanFramesPerSource",
        "Mean frames heard per source.",
        "obs:ObsAvgFramesPerSource"),
    "AvgRxPacketsPerFlow": (
        None,
        "Duplicate of AvgTxPacketsPerFlow (0 differing rows in 40,000).",
        "obs:ObsAvgFramesPerSource"),
}

# Category, for the paper's Section 3.4 table.
CATEGORY = {
    "TcMessageRate": "Control plane",
    "AverageAdvertisedLinksPerTCMessage": "Control plane",
    "AdvertisedLinksPerKnownNode": "Control plane",
    "MeanTcHopCount": "Control plane",
    "MidMessageRate": "Control plane",
    "HnaMessageRate": "Control plane",
    "RoutingOverheadBytesRatio": "Control plane",
    "ControlToDataFrameRatio": "Control plane",
    "SniffedFrameRate": "Observed traffic",
    "ObservedDataRate": "Observed traffic",
    "MeanSniffedFrameSize": "Observed traffic",
    "MeanPerSourceDataRate": "Per source",
    "PerSourceRateStd": "Per source",
    "MeanSourceVisibilityWindow": "Per source",
    "SourceVisibilityWindowStd": "Per source",
    "MeanBytesPerSource": "Per source",
    "MeanFramesPerSource": "Per source",
}


def paper_name(canonical):
    """Paper-facing name for a canonical column, or None if it is a duplicate."""
    entry = PAPER_NAMES.get(canonical)
    return entry[0] if entry else canonical


def reported_features():
    """The distinct features the paper reports, in canonical order."""
    seen, out = [], []
    for _c, (name, _d, _e) in PAPER_NAMES.items():
        if name and name not in seen:
            seen.append(name)
            out.append(name)
    return out


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from arm_spec import LISTENER

    missing = sorted(set(LISTENER) - set(PAPER_NAMES))
    extra = sorted(set(PAPER_NAMES) - set(LISTENER))
    if missing or extra:
        print("MISMATCH vs arm_spec.LISTENER")
        if missing:
            print("  missing:", missing)
        if extra:
            print("  extra  :", extra)
        sys.exit(1)

    reported = reported_features()
    print("canonical listener columns : %d" % len(LISTENER))
    print("distinct reported features : %d" % len(reported))
    print("duplicates not reported    : %d" %
          sum(1 for v in PAPER_NAMES.values() if v[0] is None))
    print()
    print("%-38s %-34s %s" % ("canonical", "paper name", "category"))
    print("-" * 100)
    for c in PAPER_NAMES:
        n = PAPER_NAMES[c][0]
        print("%-38s %-34s %s" % (c, n or "(duplicate, not reported)",
                                  CATEGORY.get(n, "") if n else ""))
