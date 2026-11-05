"""Whole-node temporal diagnostic; annotation spans are not readability labels."""
import math

def _checked(nodes, query_time):
    if not math.isfinite(query_time) or query_time < 0:
        raise ValueError('finite causal query required')
    for n in nodes:
        if not (0 <= n['begin'] <= n['end'] <= query_time):
            raise ValueError('future or invalid node')
        if type(n['tokens']) is not int or not 1 <= n['tokens'] <= 432:
            raise ValueError('whole node token budget')
    if any(a['end'] >= b['begin'] for a,b in zip(nodes,nodes[1:])):
        raise ValueError('ordered disjoint nodes required')

def choose(history, recent, intervals, query_time, *, max_nodes=2, budget=432):
    """Evaluation-only, not automatic retrieval. Never accepts correct answers."""
    _checked(history+recent,query_time)
    if max_nodes != 2 or budget != 432:
        raise ValueError('frozen first-version limits')
    if len(recent)!=4 or any(n['tokens']!=108 or not n['exact_pair'] for n in recent):
        raise ValueError('four untouched true recent pairs required')
    for a,b in intervals:
        if not all(math.isfinite(v) for v in (a,b)) or not 0<=a<=b<=query_time:
            raise ValueError('evidence interval outside causal prefix')
    # Interval intersection is only a provenance proxy, not visual support.
    hit=lambda n: any(max(n['begin'],a)<=min(n['end'],b) for a,b in intervals)
    candidates=[i for i,n in enumerate(history) if hit(n)]
    selected=[];tokens=0
    # Original chronological order, not answer-dependent ranking.
    for i in candidates:
        if len(selected)<max_nodes and tokens+history[i]['tokens']<=budget:
            selected.append(i);tokens+=history[i]['tokens']
    alternatives=[i for i,n in enumerate(history) if not hit(n)]
    control=alternatives[:len(selected)]
    matched=len(control)==len(selected) and sum(history[i]['tokens'] for i in control)==tokens
    return dict(selected_history_indices=selected, control_history_indices=control if matched else None,
        control_token_matched=matched, historical_tokens=tokens, total_visual_tokens=432+tokens,
        intersecting_history_nodes=len(candidates), omitted_matching_nodes=len(candidates)-len(selected),
        recent_interval_hit=any(hit(n) for n in recent),
        any_interval_hit=bool(candidates) or any(hit(n) for n in recent),
        selected_nodes_compressed=[not history[i]['exact_pair'] for i in selected],
        source_interval_hit_is_not_readability=True, automatic_selector=False,
        no_match_is_not_recent_sufficiency=True)

def subset_columns(full_lengths, selected_indices):
    """Preserve full-context coordinates; compact only the cache-index axis."""
    if selected_indices!=sorted(set(selected_indices)):
        raise ValueError('unique ordered node indices required')
    if any(type(i)!=int or i<0 or i>=len(full_lengths) for i in selected_indices):
        raise ValueError('node index outside full bank')
    starts=[];cursor=0
    for length in full_lengths:
        if type(length)!=int or length<=0:raise ValueError('positive full node lengths')
        starts.append(cursor);cursor+=length
    return [j for i in selected_indices for j in range(starts[i],starts[i]+full_lengths[i])]
