import os
import numpy as np
from typing import List, Optional, Tuple, Dict
from PIL import Image, ImageDraw


def _tile_canvas_size(N: int, ncols: int, tile: int, pad: int):
    nrows = int(np.ceil(N / ncols))
    W = ncols * tile + (ncols + 1) * pad
    H = nrows * tile + (nrows + 1) * pad
    return W, H, nrows


def _idx_to_rc(i: int, ncols: int, nrows: int):
    # col-major
    return i % nrows, i // nrows

def _cell_rect(r: int, c: int, tile: int, pad: int):
    x0 = pad + c * (tile + pad)
    y0 = pad + r * (tile + pad)
    x1 = x0 + tile
    y1 = y0 + tile
    return x0, y0, x1, y1

def _draw_separators_colmajor(
    draw: ImageDraw.ImageDraw,
    blocks: List[Tuple[int, int]],
    ncols: int,
    nrows: int,
    tile: int,
    pad: int,
    y_offset: int,
    N: int,
    color=(255, 255, 0),
    width: int = 4,
):
    """
    Draw separators between consecutive blocks (col-major layout).
    Rule for SAME-COLUMN boundary (horizontal separator):
      - horizontal only spans the current column
      - left endpoint draws vertical downward to bottom
      - right endpoint draws vertical upward to top
      - if an endpoint lies on outermost border, skip that vertical
    Rule for CROSS-COLUMN boundary:
      - draw a vertical separator at the column boundary (full height),
        unless it is on the outermost border.
    """
    if len(blocks) <= 1 or N <= 0:
        return

    gap = max(1, pad // 2)

    def x_left(c: int) -> int:
        return pad + c * (tile + pad)

    def x_right(c: int) -> int:
        return pad + c * (tile + pad) + tile

    def y_top(r: int) -> int:
        return pad + r * (tile + pad)

    def y_bottom(r: int) -> int:
        return pad + r * (tile + pad) + tile

    # last occupied cell by display index (N-1)
    r_last = (N - 1) % nrows
    c_last = (N - 1) // nrows

    grid_left = x_left(0)
    grid_right = x_right(c_last)
    grid_top = y_top(0) + y_offset
    # full grid height (consistent visual), not clipped to r_last
    grid_bottom = y_bottom(nrows - 1) + y_offset

    for bi in range(len(blocks) - 1):
        end_idx = blocks[bi][1]
        next_idx = end_idx + 1
        if next_idx >= N:
            continue

        r_end = end_idx % nrows
        c_end = end_idx // nrows
        r_next = next_idx % nrows
        c_next = next_idx // nrows

        if c_next == c_end:
            # ---- SAME COLUMN: horizontal boundary ----
            # horizontal y between row r_end and r_end+1
            y = y_bottom(r_end) + y_offset + gap

            # horizontal spans only this column
            hx0 = x_left(c_end)
            hx1 = x_right(c_end)

            # draw horizontal
            draw.line([(hx0, y), (hx1, y)], fill=color, width=width)

            # left vertical: from y down to bottom (unless at outermost left border)
            if hx0 > grid_left:
                xL = hx0 - gap
                draw.line([(xL, y), (xL, grid_bottom)], fill=color, width=width)

            # right vertical: from top down to y (unless at outermost right border)
            if hx1 < grid_right:
                xR = hx1 + gap
                draw.line([(xR, grid_top), (xR, y)], fill=color, width=width)

        else:
            # ---- CROSSED INTO NEXT COLUMN: vertical boundary ----
            # boundary at right edge of column c_end
            vx = x_right(c_end) + gap

            # if boundary is at the outermost right, skip
            if vx >= grid_right + pad:
                continue

            draw.line([(vx, grid_top), (vx, grid_bottom)], fill=color, width=width)

def _order_row1(paths: List[str], true_labels: Optional[np.ndarray]) -> List[int]:
    # keep dataset read-in order
    return list(range(len(paths)))

def _blocks_from_sequence(seq: List[int]) -> List[Tuple[int, int]]:
    """
    seq is a list of group IDs aligned with DISPLAY ORDER.
    Return blocks as consecutive equal segments: [(start,end),...]
    """
    blocks = []
    if len(seq) == 0:
        return blocks
    start = 0
    prev = seq[0]
    for k in range(1, len(seq)):
        if seq[k] != prev:
            blocks.append((start, k - 1))
            start = k
            prev = seq[k]
    blocks.append((start, len(seq) - 1))
    return blocks

def _order_row2(
    paths: List[str],
    cluster_ids: np.ndarray,
    true_labels: Optional[np.ndarray],
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    Row2 ordering:
    - group by cluster_id into blocks
    - order blocks by:
        1) dominant true_label earliest in dataset (approx "read-in/label order")
        2) if same dominant label: larger dominant count first
        3) then larger cluster size
        4) then cluster_id for stability
    - within a block:
        dominant-label samples first, then others; stable by path
    Returns:
      idx_row2: indices in display order
      blocks: [(start,end),...] each cluster is one block
    """
    N = len(paths)
    clusters: Dict[int, List[int]] = {}
    for i in range(N):
        c = int(cluster_ids[i])
        clusters.setdefault(c, []).append(i)

    # helper: earliest position of each true_label in read-in order
    if true_labels is not None:
        y = true_labels.astype(np.int64)
        first_pos: Dict[int, int] = {}
        for i in range(N):
            lab = int(y[i])
            if lab not in first_pos:
                first_pos[lab] = i

    block_meta = []
    for c, mem in clusters.items():
        size = len(mem)

        if true_labels is not None:
            labs = y[np.array(mem, dtype=np.int64)]
            counts = np.bincount(labs) if labs.size > 0 else np.array([0])
            maj = int(counts.argmax()) if counts.size > 0 else 10**9
            maj_cnt = int(counts[maj]) if counts.size > 0 else 0
            maj_first = int(first_pos.get(maj, 10**9))
        else:
            maj = 10**9
            maj_cnt = 0
            maj_first = 10**9

        # sort key:
        #   maj_first (earlier label appears earlier),
        #   maj (tie-break),
        #   -maj_cnt (more dominant count first),
        #   -size (bigger cluster first),
        #   c (stable)
        block_meta.append((maj_first, maj, -maj_cnt, -size, c))

    block_meta.sort()
    ordered_clusters = [c for (_, _, _, _, c) in block_meta]

    idx_row2: List[int] = []
    blocks: List[Tuple[int, int]] = []
    cur = 0

    for c in ordered_clusters:
        mem = clusters[c]

        if true_labels is not None:
            labs = y[np.array(mem, dtype=np.int64)]
            counts = np.bincount(labs) if labs.size > 0 else np.array([0])
            maj = int(counts.argmax()) if counts.size > 0 else 10**9

            # dominant-label first, keep READ-IN ORDER
            mem_maj = [i for i in mem if int(y[i]) == maj]
            mem_other = [i for i in mem if int(y[i]) != maj]

            mem_maj = sorted(mem_maj)                 # by index = read-in order
            mem_other = sorted(mem_other, key=lambda i: (int(y[i]), i))

            mem_sorted = mem_maj + mem_other
        else:
            mem_sorted = sorted(mem)


        start = cur
        idx_row2.extend(mem_sorted)
        cur += len(mem_sorted)
        end = cur - 1
        if end >= start:
            blocks.append((start, end))

    return idx_row2, blocks

def save_two_row_figure(
    paths: List[str],
    cluster_ids: np.ndarray,
    true_labels: Optional[np.ndarray],
    out_path: str,
    ncols: int = 20,
    tile: int = 64,
    pad: int = 2,
    row_gap: int = 12,
    box_color=(255, 255, 0),
):
    """
    Row1: originals in GT big-class order if true_labels exist.
    Row2: clustered, clusters ordered to match GT big-class order as much as possible.
    Yellow boxes (or box_color) mark each predicted cluster block.
    """
    N = len(paths)
    assert len(cluster_ids) == N

    idx_row1 = _order_row1(paths, true_labels)
    idx_row2, blocks = _order_row2(paths, cluster_ids, true_labels)
    
    width = max(2, tile // 32)
    
    W, Hgrid, nrows = _tile_canvas_size(N, ncols, tile, pad)
    Htotal = Hgrid * 2 + row_gap

    canvas = Image.new("RGB", (W, Htotal), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    def paste_row(idxs: List[int], y_offset: int):
        for k, i in enumerate(idxs):
            r, c = _idx_to_rc(k, ncols, nrows)
            x0, y0, x1, y1 = _cell_rect(r, c, tile, pad)
            try:
                img = Image.open(paths[i]).convert("RGB").resize((tile, tile))
            except Exception:
                img = Image.new("RGB", (tile, tile), (0, 0, 0))
            canvas.paste(img, (x0, y0 + y_offset))

    # Row 1
    paste_row(idx_row1, y_offset=0)

    # Row1: separators by true_label (only if labeled)
    if true_labels is not None:
        seq1 = [int(true_labels[i]) for i in idx_row1]
        blocks1 = _blocks_from_sequence(seq1)
        _draw_separators_colmajor(
            draw, blocks1,
            ncols=ncols, nrows=nrows, tile=tile, pad=pad, y_offset=0,
            N=N,
            color=box_color, width=width
        )
        
    # Row 2
    y2 = Hgrid + row_gap
    paste_row(idx_row2, y_offset=y2)

    # Row2: separators between cluster blocks
    _draw_separators_colmajor(
        draw, blocks,
        ncols=ncols, nrows=nrows, tile=tile, pad=pad, y_offset=y2,
        N=N,
        color=box_color, width=width
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path)
