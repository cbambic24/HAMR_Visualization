"""
Calibrate how many AMR blocks a single MPI rank can hold in memory.

The analysis scripts read a dump, then build stress-energy tensors, dyadics
and opacities on top of it. Those intermediates cost far more memory than the
block data itself, so the number of blocks a rank can safely batch is set by
the *inflated* footprint, not by the raw field sizes. This module measures
that inflation on a real dump at job start and divides it into a real memory
budget.

Nothing here knows any physics. The compute step is supplied by the caller as
a callable, so that the analysis stays in the analysis scripts and this file
stays a memory measurement.

Two deliberate choices worth knowing about:

  * Memory is measured as **RSS**, not with `tracemalloc`. tracemalloc only
    sees the Python allocator, and the tensor and opacity routines run
    through the Cython/C++ layer where the allocations are `new`, not
    PyObject. A tracemalloc-based budget would miss most of the very
    overhead this module exists to measure.

  * The peak numbers come from `ru_maxrss`, which is a monotonic high-water
    mark for the process. That makes the measurements immune to the
    allocator declining to return freed pages to the OS, but it also means
    calibration must run *early*, before the job allocates anything larger
    than the probe. See `measure_baseline_bytes`.
"""

import os
import gc
import resource
import threading
import numpy as np
from mpi4py import MPI
import hamr_read as hr

PAGE_SIZE = os.sysconf('SC_PAGE_SIZE')

# Fraction of the remaining per-rank budget held back to cover error in the
# measured multiplier. This is *not* a blanket safety factor: the fixed cost
# of the interpreter and its extensions is subtracted separately, as a
# measured constant, so this margin applies only to what is genuinely
# uncertain.
DEFAULT_MARGIN = 0.10

def get_rss_bytes():
    """Current resident set size of this process, in bytes."""
    with open('/proc/self/statm') as fin:
        return int(fin.read().split()[1]) * PAGE_SIZE

def get_peak_rss_bytes():
    """High-water mark of this process's resident set size, in bytes.
    Monotonic: it never falls, even after the memory is freed."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024

def get_mem_available_bytes():
    """Memory the kernel believes is available on this node, in bytes.
    MemAvailable rather than MemFree, so reclaimable page cache counts."""
    with open('/proc/meminfo') as fin:
        for line in fin:
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) * 1024
    raise RuntimeError('MemAvailable not present in /proc/meminfo')

def get_ranks_per_node(comm):
    """Number of MPI ranks sharing this node's memory. These ranks compete
    for one MemAvailable, so the budget has to be split between them."""
    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    ranks     = node_comm.Get_size()
    node_comm.Free()
    return ranks

class sample_peak_rss():
    """Context manager giving the peak RSS reached inside its body.

    `ru_maxrss` is monotonic over the life of the process, so once the read
    has set a high-water mark it cannot say what a later step reached on its
    own -- the multiplier comes out as exactly 1.0. Polling the current RSS
    in a background thread attributes a peak to one region of code.
    """
    def __init__(self, interval=0.005):
        self.interval = interval
        self.peak     = 0
        self._stop    = threading.Event()

    def _poll(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, get_rss_bytes())
            self._stop.wait(self.interval)

    def __enter__(self):
        self.peak    = get_rss_bytes()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info):
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, get_rss_bytes())
        return False

def count_blocks_read(H):
    """Number of blocks actually loaded into H.

    Not `H.params['n_active_total']` -- that is the dump's global block
    count and takes no notice of start_block/end_block, so using it makes the
    per-block cost too small by the ratio of the two.
    """
    for key in ('rho', 'u_gas', 'Te'):
        arr = H.data.get(key)
        if isinstance(arr, np.ndarray) and arr.ndim == 4:
            return int(arr.shape[0])
    return int(H.params['n_active_total'])

def measure_baseline_bytes():
    """Fixed per-process cost of the interpreter, numpy, mpi4py and the
    compiled extensions, in bytes.

    Call this once, immediately after imports and before any dump is read.
    Everything else in this module is expressed relative to it, and the
    monotonic peak counters make the ordering matter."""
    gc.collect()
    return get_peak_rss_bytes()

def measure_raw_read_bytes(data_dir, file_num, read_args, baseline_bytes):
    """Peak RSS added by reading one dump's grid and primitives, with no
    physics applied at all.

    Returns (bytes_above_baseline, n_blocks_read). Deliberately free of any
    analysis: the result is a pure data-size number, so it can serve as the
    denominator of a multiplier whose numerator does include the physics.
    """
    gc.collect()
    with sample_peak_rss() as probe:
        H = hr.Read(data_dir, **read_args)
        H.read_grid(file_num, direct=True)
        H.read_prims(file_num, skip_read_params=True)
        n_blocks = count_blocks_read(H)
    del H
    gc.collect()
    return probe.peak - baseline_bytes, n_blocks

def measure_compute_multiplier(data_dir, file_num, read_args, compute_fn,
                               baseline_bytes, **kwargs):
    """Factor by which the analysis inflates memory beyond the raw fields.

    Reads a probe batch of blocks, records the peak, hands the reader to
    `compute_fn`, and records the peak again. `compute_fn(H, **kwargs)` is
    whatever the calling script does to a batch -- tensor contractions,
    dyadics, opacities -- and its return value is ignored. Both peaks are
    taken above the same baseline, so

        multiplier = (read + compute) / read

    is exactly the inflation caused by the intermediates.

    Because ru_maxrss is monotonic this is safe against the allocator
    holding on to freed pages, but it does require that nothing larger has
    run in this process beforehand.
    """
    gc.collect()
    with sample_peak_rss() as read_probe:
        H = hr.Read(data_dir, **read_args)
        H.read_grid(file_num, direct=True)
        H.read_prims(file_num, skip_read_params=True)
        n_blocks = count_blocks_read(H)
    raw_bytes = read_probe.peak - baseline_bytes

    with sample_peak_rss() as compute_probe:
        compute_fn(H, **kwargs)
    total_bytes = max(compute_probe.peak, read_probe.peak) - baseline_bytes

    del H
    gc.collect()

    if raw_bytes <= 0:
        raise RuntimeError('Raw read measured as %d bytes above baseline; '
                           'calibration must run before anything larger.'
                           %raw_bytes)
    return total_bytes / raw_bytes, raw_bytes, n_blocks

def calibrate_compute_multiplier(existing_full_peak_bytes, data_dir, file_num,
                                 read_args, baseline_bytes):
    """Multiplier inferred from a previous full-dump peak instead of by
    running the physics. Use when no `compute_fn` is available.

    The denominator is measured with the *same* read mode as whatever peak
    is passed in -- non-block here, matching a full-dump run. Measuring it
    by-block instead would fold the upscaling saving into the multiplier,
    and then double-count it when the multiplier is applied back to a
    by-block measurement.
    """
    args = dict(read_args)
    args['by_block'] = 0
    args.pop('start_block', None)
    args.pop('end_block', None)
    raw_bytes, _ = measure_raw_read_bytes(data_dir, file_num, args,
                                          baseline_bytes)
    return existing_full_peak_bytes / raw_bytes

def get_per_rank_budget(comm, baseline_bytes, margin=DEFAULT_MARGIN,
                        ranks_per_node=None):
    """Bytes this rank may spend on block data, in bytes.

    MemAvailable is split between the ranks sharing the node. The fixed cost
    of the interpreter and its extensions is then subtracted as a measured
    constant rather than folded into a percentage, so that `margin` covers
    only uncertainty in the multiplier.

    `ranks_per_node` may be supplied by a caller that has already computed
    it. That is not an optimisation: get_ranks_per_node is COLLECTIVE, so a
    caller running this on one rank only must pass the value in rather than
    let this function derive it. See plan_block_batches.
    """
    if ranks_per_node is None:
        ranks_per_node = get_ranks_per_node(comm)
    per_rank = get_mem_available_bytes() / ranks_per_node
    return max((per_rank - baseline_bytes) * (1.0 - margin), 0.0)

def determine_block_batch_size(data_dir, file_num, read_args, comm,
                               baseline_bytes, compute_fn=None,
                               compute_multiplier=None, n_probe=8,
                               margin=DEFAULT_MARGIN, **kwargs):
    """Number of blocks this rank can read and analyze at once.

    Supply either `compute_fn`, in which case the multiplier is measured
    directly by running it on the probe batch, or `compute_multiplier` if it
    has already been calibrated. Returns a dict of everything measured, so
    the caller can log the reasoning rather than just the answer.
    """
    args = dict(read_args)
    args['by_block']    = 1
    args['start_block'] = 0
    args['end_block']   = n_probe

    if compute_fn is not None:
        multiplier, raw_bytes, n_blocks = measure_compute_multiplier(
            data_dir, file_num, args, compute_fn, baseline_bytes, **kwargs)
    elif compute_multiplier is not None:
        multiplier = compute_multiplier
        raw_bytes, n_blocks = measure_raw_read_bytes(data_dir, file_num, args,
                                                     baseline_bytes)
    else:
        raise ValueError('Supply either compute_fn or compute_multiplier.')

    per_block = (raw_bytes / max(n_blocks, 1)) * multiplier
    budget    = get_per_rank_budget(comm, baseline_bytes, margin)
    n_batch   = max(int(budget // per_block), 1) if per_block > 0 else 1

    return {
        'n_batch'            : n_batch,
        'per_block_bytes'    : per_block,
        'budget_bytes'       : budget,
        'multiplier'         : multiplier,
        'raw_probe_bytes'    : raw_bytes,
        'n_probe_blocks'     : n_blocks,
        'baseline_bytes'     : baseline_bytes,
        'ranks_per_node'     : get_ranks_per_node(comm),
        'mem_available_bytes': get_mem_available_bytes(),
    }

def measure_batch_bytes(batch_fn, n_start=8, n_max=512, rel_tol=0.35,
                        cap_bytes=None):
    """Fixed and marginal memory cost of a batch, by probing growing sizes.

    `batch_fn(n_blocks)` runs one COMPLETE batch -- read and analysis -- for
    the given number of blocks. Its return value is ignored.

    Two things make the naive estimate wrong, both in the dangerous
    direction (understating cost, hence oversized batches):

    1. A large part of the footprint does not scale with the batch. Read()
       peaks at ~640 MB on THIN_RAD_MAD_HIRES_TRACER before a single block
       is analysed -- a (nmax, 200) int32 block array plus the gd temporary
       used to build it -- and nmax counts every block the octree COULD
       create, not the nb that exist. Dividing one peak by n folds that in:
       measured there, peak/n gives 29.6 MB/block at n=32 against a true
       marginal 16.7. Hence two points and a slope.

    2. Below that floor the curve is FLAT, so a fit taken there is noise.
       Measured: peak(8) = 695.1 MB and peak(16) = 696.4 MB -- 1.3 MB apart
       for eight more blocks -- while peak(32) = 948 and peak(64) = 1499
       give a clean ~17 MB/block. The probe therefore doubles until the two
       peaks separate by rel_tol, rather than trusting a fixed pair. The
       floor scales with nmax, so no fixed pair can be right for every run.

    Peaks come from ru_maxrss rather than the sampling thread in
    sample_peak_rss: it is a kernel-tracked high-water mark and cannot miss
    a short-lived spike, and the tensor intermediates that set the peak can
    be allocated and freed inside one 5 ms poll. It is monotonic over the
    process, but the probes run in INCREASING order, so its value after each
    probe is that probe's peak, and the common baseline cancels in the slope.

    rel_tol trades accuracy for probe size. Measured on
    THIN_RAD_MAD_HIRES_TRACER against a true marginal of ~17.2 MB/block:

        rel_tol   probes    per_block   error   probe peak
          0.10    (16, 32)   14.27 MB    -17%     929 MB
          0.35    (32, 64)   16.55 MB   -3.4%    1.46 GB
          0.60    (64,128)   17.14 MB   -0.3%    2.57 GB

    0.35 is the default: the residual error is in the unsafe direction
    (batches slightly too large) but well inside DEFAULT_MARGIN, while the
    probe stays small enough not to be the largest allocation of the job.
    TRANS_XRB_HIGH_SANE settles at (8, 16) for any of these, because its
    block array is 10.5 MB and there is barely a floor to clear.

    Call this before the job allocates anything larger than the probe:
    ru_maxrss is process-wide, so a bigger earlier allocation flattens
    every probe and the fallback (correctly, but uselessly) triggers.
    """
    n_prev = int(n_start)
    gc.collect(); batch_fn(n_prev); gc.collect()
    p_prev = get_peak_rss_bytes()

    # Growing the probe costs memory, so growth stops once the observed
    # peak passes cap_bytes -- a fraction of this rank's share of
    # MemAvailable. That share needs no probe to know (MemAvailable /
    # ranks_per_node), so there is no circularity.
    #
    # NOTE this bounds growth, it does not bound the peak: the check uses
    # the PREVIOUS probe's peak, so the final probe can overshoot by up to
    # one doubling. Measured with a 1 GB cap on HIRES, the probe stopped
    # after peaking at 1.46 GB. Predicting the next peak instead does not
    # work -- the observed increments (1 MB, then 233 MB) under-predict the
    # next one badly, because the curve is convex where it leaves the
    # construction floor. Hence cap_frac defaults to 0.25 rather than 0.5,
    # so that a one-doubling overshoot still lands near half the share.
    n, p, separated = n_prev, p_prev, False
    while n_prev < n_max:
        if cap_bytes is not None and p_prev >= cap_bytes:
            break
        n = min(2 * n_prev, n_max)
        gc.collect(); batch_fn(n); gc.collect()
        p = get_peak_rss_bytes()
        if (p - p_prev) >= rel_tol * p_prev:
            separated = True
            break
        # Still on the flat part: discard the low point and climb.
        n_prev, p_prev = n, p

    fallback = None
    if not separated or n <= n_prev:
        # Never cleared the construction floor -- either the probe hit
        # n_max, or the memory cap stopped it growing. peak/n at the largest
        # probe still contains the fixed cost, so it OVERSTATES the
        # per-block price and yields smaller batches -- the safe way to be
        # wrong.
        why = 'memory cap' if (cap_bytes is not None
                               and p_prev >= cap_bytes) else 'n_max=%d' %n_max
        fallback  = ('probe never separated from the fixed cost by %.0f%% '
                     '(stopped by %s); using peak/n, which overstates '
                     'per-block cost' %(100 * rel_tol, why))
        per_block = p / float(max(n, 1))
        fixed     = 0.0
    else:
        per_block = (p - p_prev) / float(n - n_prev)
        fixed     = p_prev - n_prev * per_block
        if per_block <= 0.0:
            fallback  = ('slope %+.3e bytes/block was not positive; using '
                         'peak/n instead' %per_block)
            per_block = p / float(max(n, 1))
            fixed     = 0.0
        elif fixed < 0.0:
            fallback = ('fit gave a negative fixed cost (%.1f MB); clamped '
                        'to 0' %(fixed / 1024.0**2))
            fixed    = 0.0

    return {
        'per_block_bytes': per_block,
        'fixed_bytes'    : fixed,
        'probe_sizes'    : (n_prev, n),
        'probe_peaks'    : (p_prev, p),
        'fallback'       : fallback,
    }

def plan_block_batches(batch_fn, comm, margin=DEFAULT_MARGIN,
                       root=0, cap_frac=0.25, **probe_kwargs):
    """Blocks per batch, measured once on `root` and broadcast.

    Measured on one rank rather than all of them for three reasons: the
    probe reads the same blocks everywhere, so N measurements cost N-1
    redundant reads (each rebuilding the nmax-sized block array); every rank
    must agree or the single-file driver's collective Reduce desynchronises;
    and MemAvailable is a node-wide, time-varying quantity, so ranks
    measuring at different moments would disagree systematically rather than
    randomly -- whoever calibrates last sees the least memory.
    """
    # get_ranks_per_node is COLLECTIVE -- Split_type must be entered by every
    # rank in comm. It therefore cannot be called inside the `rank == root`
    # branch below: root would block in Split_type waiting for the others
    # while the others block in the bcast waiting for root, and the job hangs
    # with every rank at 100% CPU (MPI busy-wait) and no output. Computed
    # once, here, on all ranks, and passed down.
    ranks_per_node = get_ranks_per_node(comm)

    info = None
    if comm.Get_rank() == root:
        # Bound the probe by a fraction of this rank's memory share, which
        # is knowable without any measurement.
        probe_kwargs.setdefault(
            'cap_bytes',
            cap_frac * get_mem_available_bytes() / ranks_per_node)
        info = measure_batch_bytes(batch_fn, **probe_kwargs)
        budget = get_per_rank_budget(comm, info['fixed_bytes'], margin,
                                     ranks_per_node=ranks_per_node)
        info['budget_bytes']        = budget
        info['ranks_per_node']      = ranks_per_node
        info['mem_available_bytes'] = get_mem_available_bytes()
        info['n_batch'] = max(int(budget // info['per_block_bytes']), 1)
    return comm.bcast(info, root=root)

def summarize(info):
    """Human-readable form of `plan_block_batches`' output.

    Every field is fetched with .get and missing ones print as n/a. This is
    pure logging: it must never be the thing that ends a job. The previous
    version keyed on `determine_block_batch_size`'s dict -- baseline_bytes,
    n_probe_blocks, raw_probe_bytes, multiplier -- none of which
    measure_batch_bytes returns, so the first real calibration died inside a
    print with a KeyError after the probes had already succeeded.
    """
    GB, MB = 1024.0 ** 3, 1024.0 ** 2

    def num(key, scale, fmt='%8.2f'):
        value = info.get(key)
        return '     n/a' if value is None else fmt % (value / scale)

    lines = [
        'Block budget calibration',
        '  MemAvailable          : %s GB' % num('mem_available_bytes', GB),
        '  ranks sharing node    : %8s'   % info.get('ranks_per_node', 'n/a'),
        '  per-rank budget       : %s GB' % num('budget_bytes', GB),
        '  fixed cost per batch  : %s GB' % num('fixed_bytes', GB),
        '  cost per block        : %s MB' % num('per_block_bytes', MB),
    ]

    # The probe pair is the evidence behind per_block_bytes: two peaks and
    # the slope between them. Worth logging, because a pair that is too
    # close together means the fit was taken on the flat part of the curve.
    sizes, peaks = info.get('probe_sizes'), info.get('probe_peaks')
    if sizes and peaks:
        lines.append('  probe sizes (blocks)  : %8s' % ('%d, %d' % tuple(sizes)))
        lines.append('  probe peaks           : %8s'
                     % ('%.2f, %.2f GB' % (peaks[0] / GB, peaks[1] / GB)))

    if info.get('n_batch_per_rank') is not None:
        lines.append('  blocks/batch per rank : %8d' % info['n_batch_per_rank'])
    lines.append('  --> blocks per batch  : %8s' % info.get('n_batch', 'n/a'))
    if info.get('fallback'):
        lines.append('  FALLBACK              : %s' % info['fallback'])
    return '\n'.join(lines)
