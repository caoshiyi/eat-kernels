"""Host planner for the CP fixup grid.  The tail of every head is cut into m consecutive dual pieces (tile-aligned: pt = piece start tiles
of 128 tokens, pt[m] = tiles per head).  Head h is OWNED by CTA h mod G: the CTA runs the state chain of h (m items, each priced `sw`
tiles) and then the output tiles of h's pieces 1..m-1 (their entering states S_in are computed by that same CTA - no inter-CTA
dependency).  The piece-0 tiles (state S_A, head-major) are cut into contiguous per-CTA ranges that balance the owners' fixed work.
Returns (seg, seg_ptr): seg = flat int32 [nseg][4] of (head, piece, first tile of the head's tail, tile count); seg_ptr per CTA."""


def fix_schedule(H, tail, G, sw, pt=None, sw1=None):
    tph = (tail + 127) // 128
    if pt is None:
        pt = [0, tph]
    m = len(pt) - 1
    if sw1 is None:
        sw1 = sw
    n0 = pt[1] * H                                   # piece-0 tiles (S_A as the operand)
    rest = tph - pt[1]                               # tiles of pieces >= 1 per head
    owned = [list(range(b, H, G)) for b in range(G)]
    fixed = [len(o) * (sw + (m - 1) * sw1 + rest) for o in owned]
    U = n0 + sum(fixed)
    seg, ptr = [], [0]
    cum_fixed, start = 0.0, 0
    for b in range(G):
        for h in owned[b]:
            for k in range(1, m):
                if pt[k + 1] > pt[k]:
                    seg += [h, k, pt[k], pt[k + 1] - pt[k]]
        cum_fixed += fixed[b]
        end = int(round((b + 1) * U / G - cum_fixed))
        end = min(max(end, start), n0)
        if b == G - 1:
            end = n0
        u = start
        while u < end:                               # head-major piece-0 tiles [start, end) cut at head boundaries
            h = u // pt[1]
            e = min(end, (h + 1) * pt[1])
            seg += [h, 0, u - h * pt[1], e - u]
            u = e
        start = end
        ptr.append(len(seg) // 4)
    return seg, ptr
