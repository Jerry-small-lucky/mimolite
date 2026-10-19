# v8 函数级 pass@1 vs pass@8 对比（评测 100 题）

| 函数 | 题数 | pass@1 通过 | pass@8 通过 | 类型 |
|---|---:|---:|---:|---|
| factorial | 16 | 12 | 15 | 部分通过 |
| count_characters | 8 | 8 | 8 | 已稳（p1 全过） |
| count_ones | 5 | 3 | 5 | p8 补全（p8 全过） |
| count_chars | 4 | 0 | 1 | 多采样命中（p1=0→p8>0） |
| process_string | 4 | 0 | 4 | 多采样命中（p1=0→p8>0） |
| count_elements | 4 | 2 | 4 | p8 补全（p8 全过） |
| fib | 4 | 0 | 4 | 多采样命中（p1=0→p8>0） |
| merge | 3 | 0 | 0 | 双零死题 |
| reverse_linked_list | 3 | 0 | 1 | 多采样命中（p1=0→p8>0） |
| count_words | 3 | 0 | 1 | 多采样命中（p1=0→p8>0） |
| sum_array | 2 | 0 | 2 | 多采样命中（p1=0→p8>0） |
| sum_even_numbers | 2 | 0 | 1 | 多采样命中（p1=0→p8>0） |
| convert_to_ascii | 2 | 0 | 2 | 多采样命中（p1=0→p8>0） |
| count_set_bits | 2 | 0 | 2 | 多采样命中（p1=0→p8>0） |
| factor_sum | 2 | 0 | 0 | 双零死题 |
| count_frequencies | 1 | 0 | 0 | 双零死题 |
| count_ways | 1 | 1 | 1 | 已稳（p1 全过） |
| merge_sorted_arrays | 1 | 0 | 0 | 双零死题 |
| count_char_occurrences | 1 | 0 | 0 | 双零死题 |
| count_even_and_sum | 1 | 1 | 0 | 已稳（p1 全过） |
| process_array | 1 | 0 | 0 | 双零死题 |
| find_top_k | 1 | 0 | 0 | 双零死题 |
| multiply | 1 | 0 | 1 | 多采样命中（p1=0→p8>0） |
| count_between | 1 | 0 | 0 | 双零死题 |
| count_one_occurrences | 1 | 0 | 0 | 双零死题 |
| sort_and_check | 1 | 0 | 0 | 双零死题 |
| max_activities | 1 | 0 | 0 | 双零死题 |
| count_even_numbers | 1 | 0 | 1 | 多采样命中（p1=0→p8>0） |
| is_adjacent_diff_one | 1 | 0 | 0 | 双零死题 |
| calculate_stats | 1 | 0 | 0 | 双零死题 |
| compare_lists | 1 | 0 | 0 | 双零死题 |
| count_frequency | 1 | 0 | 0 | 双零死题 |
| convert_to_upper_without_spaces | 1 | 0 | 0 | 双零死题 |
| count_occurrences | 1 | 0 | 0 | 双零死题 |
| remove_duplicates | 1 | 0 | 0 | 双零死题 |
| sort_list | 1 | 1 | 1 | 已稳（p1 全过） |
| count_most_frequent | 1 | 0 | 0 | 双零死题 |
| is_palindrome | 1 | 1 | 1 | 已稳（p1 全过） |
| sum_list | 1 | 0 | 0 | 双零死题 |
| maxNonOverlapping | 1 | 0 | 0 | 双零死题 |
| calculate_even_stats | 1 | 0 | 0 | 双零死题 |
| is_equal_min_max | 1 | 0 | 0 | 双零死题 |
| merge_two_sorted_arrays | 1 | 0 | 0 | 双零死题 |
| count_even | 1 | 0 | 1 | 多采样命中（p1=0→p8>0） |
| is_all_equal | 1 | 0 | 0 | 双零死题 |
| stats_array | 1 | 0 | 0 | 双零死题 |
| is_strictly_increasing | 1 | 0 | 0 | 双零死题 |
| maxNonOverlappingIntervals | 1 | 0 | 0 | 双零死题 |
| flip_bit | 1 | 0 | 1 | 多采样命中（p1=0→p8>0） |
| sum_even | 1 | 0 | 0 | 双零死题 |
| flip_bits | 1 | 0 | 0 | 双零死题 |

## 汇总：稳定 5 个函数 / 多采样命中 13 个 / 部分 3 个 / 双零死题 30 个

### 多采样命中型（模型已学到，单采样确定性不足——数据侧价值的直接证据）
| 函数 | 题数 | p8 命中 |
|---|---:|---:|
| count_chars | 4 | 1 |
| process_string | 4 | 4 |
| fib | 4 | 4 |
| reverse_linked_list | 3 | 1 |
| count_words | 3 | 1 |
| sum_array | 2 | 2 |
| sum_even_numbers | 2 | 1 |
| convert_to_ascii | 2 | 2 |
| count_set_bits | 2 | 2 |
| multiply | 1 | 1 |
| count_even_numbers | 1 | 1 |
| count_even | 1 | 1 |
| flip_bit | 1 | 1 |

### 双零死题（能力未达 / 评测设计问题）
| 函数 | 题数 |
|---|---:|
| merge | 3 |
| factor_sum | 2 |
| count_frequencies | 1 |
| merge_sorted_arrays | 1 |
| count_char_occurrences | 1 |
| process_array | 1 |
| find_top_k | 1 |
| count_between | 1 |
| count_one_occurrences | 1 |
| sort_and_check | 1 |
| max_activities | 1 |
| is_adjacent_diff_one | 1 |
| calculate_stats | 1 |
| compare_lists | 1 |
| count_frequency | 1 |
| convert_to_upper_without_spaces | 1 |
| count_occurrences | 1 |
| remove_duplicates | 1 |
| count_most_frequent | 1 |
| sum_list | 1 |
| maxNonOverlapping | 1 |
| calculate_even_stats | 1 |
| is_equal_min_max | 1 |
| merge_two_sorted_arrays | 1 |
| is_all_equal | 1 |
| stats_array | 1 |
| is_strictly_increasing | 1 |
| maxNonOverlappingIntervals | 1 |
| sum_even | 1 |
| flip_bits | 1 |