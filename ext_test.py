import cpp_tokenizer as ct

import time

vocab = {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "[MASK]": 3, "[PAD]": 4, "ipv-首猜": 5, "ipv-搜索": 6, "ipv-详猜": 7,
         "买家聊天-搜索": 8,
         "卖家聊天-搜索": 9, "赚闲鱼币-闲鱼币": 10, "ipv-同城": 11, "push点击-push": 12, "dpv-会玩": 13, "卖家发布-default": 14, "买家聊天-详猜": 15,
         "买家聊天-首猜": 16, "卖家聊天-详猜": 17, "卖家聊天-首猜": 18, "买家成交-搜索": 19, "卖家成交-搜索": 20, "消费闲鱼币-闲鱼币": 21, "卖家任务-default": 22,
         "回收估价-省心卖": 23, "ipv-关注tab": 24, "卖家聊天-同城": 25, "买家聊天-同城": 26, "买家成交-详猜": 27, "卖家成交-详猜": 28, "买家成交-首猜": 29,
         "寄卖估价-省心卖": 30, "卖家成交-首猜": 31, "被赞-会玩": 32, "赞-会玩": 33, "回收下单-省心卖": 34, "ipv-优品": 35, "发布帖子-会玩": 36,
         "回收成交-省心卖": 37, "评论-会玩": 38, "回收寄出-省心卖": 39, "被评论-会玩": 40, "买家聊天-关注tab": 41, "买家聊天-优品": 42, "卖家聊天-关注tab": 43,
         "卖家成交-同城": 44, "寄卖寄出-省心卖": 45, "卖家聊天-优品": 46, "买家成交-同城": 47, "寄卖下单-省心卖": 48, "寄卖成交-省心卖": 49, "买家成交-关注tab": 50,
         "卖家成交-关注tab": 51, "卖家成交-优品": 52}

feature_arr = ["0;买家聊天-详猜;1|1;ipv-详猜;1|2;卖家聊天-首猜;1|3;卖家聊天-搜索;3",
               "0;买家聊天-首猜;1|1;ipv-详猜;1|28;卖家任务-default;2|28;卖家聊天-搜索;3|29;卖家聊天-搜索;3"]

feature_arr_2 = [
                    "0;买家成交-详猜;1|0;买家聊天-详猜;1|0;ipv-详猜;1|13;卖家聊天-搜索;1|0;买家聊天-首猜;1|1;买家聊天-搜索;2|1;ipv-首猜;7|1;ipv-搜索;2|1;买家聊天-default;1|1;ipv-搜索;1|1;ipv-首猜;2|1;dpv-会玩;1|1;ipv-首猜;8|1;买家聊天-首猜;1|1;ipv-首猜;21|1;ipv-搜索;9|1;买家聊天-default;1|1;ipv-搜索;16|3;ipv-首猜;9|3;dpv-会玩;1|3;ipv-首猜;1|3;dpv-会玩;1|3;赞-会玩;1|3;dpv-会玩;1|3;ipv-首猜;2|3;ipv-搜索;2|3;ipv-首猜;6|4;ipv-搜索;5|5;ipv-搜索;1|5;买家聊天-搜索;1|5;ipv-搜索;2|5;dpv-会玩;1|6;ipv-搜索;7|6;ipv-首猜;11|7;ipv-首猜;6|7;ipv-搜索;2|7;ipv-首猜;3|8;ipv-首猜;3|8;买家聊天-default;1|9;ipv-首猜;2|9;买家聊天-default;1|9;ipv-搜索;2|9;买家聊天-搜索;1|9;ipv-搜索;1|9;买家聊天-搜索;1|9;ipv-搜索;2|9;买家聊天-搜索;1|9;ipv-搜索;13|9;买家聊天-搜索;1|9;ipv-搜索;1|9;ipv-首猜;2|9;ipv-搜索;6|9;买家聊天-default;1|10;ipv-首猜;4|10;买家聊天-搜索;1|11;ipv-首猜;7|11;买家聊天-搜索;1|11;ipv-搜索;3|11;ipv-首猜;10|12;ipv-搜索;2|12;买家聊天-首猜;1|12;ipv-首猜;11|12;ipv-搜索;2|12;ipv-首猜;6|13;ipv-首猜;1|13;买家聊天-default;1|13;买家聊天-首猜;1|13;ipv-首猜;20|14;ipv-搜索;1|14;买家聊天-搜索;1|14;ipv-搜索;1|14;买家聊天-搜索;1|14;ipv-搜索;2|14;ipv-首猜;15|14;买家聊天-搜索;1|15;ipv-搜索;9|15;ipv-首猜;28|15;dpv-会玩;1|15;ipv-首猜;1|15;dpv-会玩;1|15;ipv-首猜;13|15;买家聊天-搜索;1|16;ipv-搜索;2|16;dpv-会玩;1|16;ipv-搜索;3|16;买家聊天-搜索;1|16;ipv-搜索;14|16;ipv-首猜;13|17;ipv-搜索;1|17;买家聊天-搜索;1|17;ipv-搜索;2|17;买家聊天-搜索;1|17;ipv-搜索;4|17;买家聊天-搜索;1|17;ipv-搜索;11|17;ipv-首猜;2|18;ipv-搜索;5|18;ipv-首猜;10|18;dpv-会玩;1|18;ipv-首猜;1"] * 1000
start = time.time()
cpp_return_ndarray_obj = ct.deal_batch(feature_arr, vocab, 10)
result = {'temporal_distance_list': cpp_return_ndarray_obj[0],
          'action_id_list': cpp_return_ndarray_obj[1],
          'frequency_list': cpp_return_ndarray_obj[2],
          'attention_mask': cpp_return_ndarray_obj[3]
          }
time_cost = time.time() - start
print('batch_size=', len(feature_arr), 'time_cost (ms)', time_cost * 1000)
if len(feature_arr) < 10:
    print(result)
# True
# print(result.flags.owndata)
