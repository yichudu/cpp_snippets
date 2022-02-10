#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <sys/time.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

using namespace std;

std::vector<std::string> stringSplitForSpeed(const std::string &text, string delim)
{
    vector<string> res;
    int temp = 0;
    string::size_type pos = 0;
    while (true)
    {
        pos = text.find(delim, temp);
        if (pos == string::npos)
            break;
        res.push_back(text.substr(temp, pos - temp));
        temp = pos + delim.length();
    }
    res.push_back(text.substr(temp, text.length()));
    return res;
}

void freePointer(vector<vector<int> *> vv)
{
    for (auto x : vv)
        free(x);
}

/*
    feature: "0;买家聊天-详猜;1|0;ipv-详猜;1|0;卖家聊天-首猜;1"
    vocab: omit
    length_limit: 100

    ----------

   [temporal_distance_list,action_id_list,frequency_list,attention_mask]

*/

vector<vector<int> *> deal_one(const string &feature, map<string, int> &vocab, int length_limit)
{
    timeval tv;
    gettimeofday(&tv, NULL);
    long one_t1 = tv.tv_sec * 1000 + tv.tv_usec / 1000;

    vector<int> *temporal_distance_list = new vector<int>(length_limit);
    vector<int> *action_id_list = new vector<int>(length_limit);
    vector<int> *frequency_list = new vector<int>(length_limit);
    vector<int> *attention_mask = new vector<int>(length_limit);

    vector<std::string> elements = stringSplitForSpeed(feature, "|");

    int i = 0;
    for (auto &ele : elements)
    {
        if (i == length_limit)
            break;
        vector<std::string> parts = stringSplitForSpeed(ele, ";");
        // 非负保障
        int temporal_distance = max(0, atoi(parts[0].c_str()));
        (*temporal_distance_list)[i] = temporal_distance;

        string action_thin = parts[1];
        if (vocab.find(action_thin) != vocab.end())
            (*action_id_list)[i] = vocab[action_thin];
        else
            (*action_id_list)[i] = vocab["[UNK]"];
          
        int frequency = max(0, atoi(parts[2].c_str()));
        (*frequency_list)[i] = frequency;
        (*attention_mask)[i] = 1;
        i++;
    } // for

    // 必要时 PAD
    for (; i < length_limit; i++)
    {
        (*temporal_distance_list)[i] = 0;
        (*action_id_list)[i] = vocab["[PAD]"];
        (*frequency_list)[i] = 0;
        (*attention_mask)[i] = 0;
    }

    return vector<vector<int> *>{temporal_distance_list, action_id_list, frequency_list, attention_mask};
}

/*
[
   "0;买家聊天-详猜;1|0;ipv-详猜;1|0;卖家聊天-首猜;1|1;卖家聊天-搜索;3|1;卖家成交-搜索;2|1;卖家聊天-搜索;5|2;卖家成交-搜索;1|2;卖家聊天-搜索;5|2;卖家成交-搜索;1|2;卖家聊天-搜索;8"
    ,
    "0;买家成交-详猜;1|0;买家聊天-详猜;1|0;ipv-详猜;1|0;卖家聊天-首猜;1|1;卖家聊天-搜索;3|1;卖家成交-搜索;2|1;卖家聊天-搜索;5|2;卖家成交-搜索;1|2;卖家聊天-搜索;5|2;卖家成交-搜索;1"

]

-------------
{
    "sample_id": ["111_买家聊天-详猜|ipv-详猜", "222_买家成交-详猜|买家聊天-详猜|ipv-详猜"],
    "temporal_distance_list": [
        [0, 0, 1, 2, 3],
        [0, 0, 1, 2, 3]
    ],
    "action_id_list": [
        [3, 4, 5, 6, 0],
        [3, 4, 5, 6, 0]
    ],
    "frequency_list": [
        [2, 3, 1, 1, 1],
        [2, 3, 1, 1, 1]
    ],
    "attention_mask": [
        [1, 1, 1, 1, 0],
        [1, 1, 1, 1, 0]
    ]
}
*/
py::array_t<int> deal_batch(const vector<string> &feature_arr, map<string, int> &vocab, size_t length_limit)
{
    // 中文验证通过
    // for(auto& pair : vocab)
    //     cout<<pair.first;

    size_t batch_size = feature_arr.size();

    vector<vector<int> *> temporal_distance_list(batch_size);
    vector<vector<int> *> action_id_list(batch_size);
    vector<vector<int> *> frequency_list(batch_size);
    vector<vector<int> *> attention_mask(batch_size);

    vector<vector<vector<int> *> *> triple_v = {&temporal_distance_list, &action_id_list, &frequency_list, &attention_mask};

    double inner_time_sum = 0;
    int i = 0;
    for (auto &feature : feature_arr)
    {
        vector<vector<int> *> one_result = deal_one(feature, vocab, length_limit);
        temporal_distance_list[i] = one_result[0];
        action_id_list[i] = one_result[1];
        frequency_list[i] = one_result[2];
        attention_mask[i] = one_result[3];
        i++;
    }

    size_t int_size = sizeof(int);
    py::array::ShapeContainer shape = {triple_v.size(), batch_size, length_limit};
    py::array::StridesContainer strides = {batch_size * length_limit * int_size, length_limit * int_size, int_size};
    auto ndarray = py::array_t<int>(shape, strides);
    auto mutable_ref = ndarray.mutable_unchecked<3>();

    for (size_t i = 0; i < triple_v.size(); i++)
        for (size_t j = 0; j < batch_size; j++)
            for (size_t k = 0; k < length_limit; k++)
                mutable_ref(i, j, k) = (*(*triple_v[i])[j])[k];

    for (auto& x : triple_v)
        freePointer(*x);

    return ndarray;
}

int main()
{

    string one_feature = "0;卖家成交-详猜;1|0;卖家聊天-详猜;1|13;卖家聊天-搜索;1|0;买家聊天-首猜;1|1;买家聊天-搜索;2|1;ipv-首猜;7|1;ipv-搜索;2|1;买家聊天-default;1|1;ipv-搜索;1|1;ipv-首猜;2|1;dpv-会玩;1|1;ipv-首猜;8|1;买家聊天-首猜;1|1;ipv-首猜;21|1;ipv-搜索;9|1;买家聊天-default;1|1;ipv-搜索;16|3;ipv-首猜;9|3;dpv-会玩;1|3;ipv-首猜;1|3;dpv-会玩;1|3;赞-会玩;1|3;dpv-会玩;1|3;ipv-首猜;2|3;ipv-搜索;2|3;ipv-首猜;6|4;ipv-搜索;5|5;ipv-搜索;1|5;买家聊天-搜索;1|5;ipv-搜索;2|5;dpv-会玩;1|6;ipv-搜索;7|6;ipv-首猜;11|7;ipv-首猜;6|7;ipv-搜索;2|7;ipv-首猜;3|8;ipv-首猜;3|8;买家聊天-default;1|9;ipv-首猜;2|9;买家聊天-default;1|9;ipv-搜索;2|9;买家聊天-搜索;1|9;ipv-搜索;1|9;买家聊天-搜索;1|9;ipv-搜索;2|9;买家聊天-搜索;1|9;ipv-搜索;13|9;买家聊天-搜索;1|9;ipv-搜索;1|9;ipv-首猜;2|9;ipv-搜索;6|9;买家聊天-default;1|10;ipv-首猜;4|10;买家聊天-搜索;1|11;ipv-首猜;7|11;买家聊天-搜索;1|11;ipv-搜索;3|11;ipv-首猜;10|12;ipv-搜索;2|12;买家聊天-首猜;1|12;ipv-首猜;11|12;ipv-搜索;2|12;ipv-首猜;6|13;ipv-首猜;1|13;买家聊天-default;1|13;买家聊天-首猜;1|13;ipv-首猜;20|14;ipv-搜索;1|14;买家聊天-搜索;1|14;ipv-搜索;1|14;买家聊天-搜索;1|14;ipv-搜索;2|14;ipv-首猜;15|14;买家聊天-搜索;1|15;ipv-搜索;9|15;ipv-首猜;28|15;dpv-会玩;1|15;ipv-首猜;1|15;dpv-会玩;1|15;ipv-首猜;13|15;买家聊天-搜索;1|16;ipv-搜索;2|16;dpv-会玩;1|16;ipv-搜索;3|16;买家聊天-搜索;1|16;ipv-搜索;14|16;ipv-首猜;13|17;ipv-搜索;1|17;买家聊天-搜索;1|17;ipv-搜索;2|17;买家聊天-搜索;1|17;ipv-搜索;4|17;买家聊天-搜索;1|17;ipv-搜索;11|17;ipv-首猜;2|18;ipv-搜索;5|18;ipv-首猜;10|18;dpv-会玩;1|18;ipv-首猜;1|18;dpv-会玩;1";

    map<string, int> m = {{"[UNK]", 0}, {"[CLS]", 1}, {"[SEP]", 2}, {"[MASK]", 3}, {"[PAD]", 4}, {"ipv-首猜", 5}, {"ipv-搜索", 6}, {"ipv-详猜", 7}, {"买家聊天-搜索", 8}, {"卖家聊天-搜索", 9}, {"赚闲鱼币-闲鱼币", 10}, {"ipv-同城", 11}, {"push点击-push", 12}, {"dpv-会玩", 13}, {"卖家发布-default", 14}, {"买家聊天-详猜", 15}, {"买家聊天-首猜", 16}, {"卖家聊天-详猜", 17}, {"卖家聊天-首猜", 18}, {"买家成交-搜索", 19}, {"卖家成交-搜索", 20}, {"消费闲鱼币-闲鱼币", 21}, {"卖家任务-default", 22}, {"回收估价-省心卖", 23}, {"ipv-关注tab", 24}, {"卖家聊天-同城", 25}, {"买家聊天-同城", 26}, {"买家成交-详猜", 27}, {"卖家成交-详猜", 28}, {"买家成交-首猜", 29}, {"寄卖估价-省心卖", 30}, {"卖家成交-首猜", 31}, {"被赞-会玩", 32}, {"赞-会玩", 33}, {"回收下单-省心卖", 34}, {"ipv-优品", 35}, {"发布帖子-会玩", 36}, {"回收成交-省心卖", 37}, {"评论-会玩", 38}, {"回收寄出-省心卖", 39}, {"被评论-会玩", 40}, {"买家聊天-关注tab", 41}, {"买家聊天-优品", 42}, {"卖家聊天-关注tab", 43}, {"卖家成交-同城", 44}, {"寄卖寄出-省心卖", 45}, {"卖家聊天-优品", 46}, {"买家成交-同城", 47}, {"寄卖下单-省心卖", 48}, {"寄卖成交-省心卖", 49}, {"买家成交-关注tab", 50}, {"卖家成交-关注tab", 51}, {"卖家成交-优品", 52}};

    auto batch_size = 1000;
    vector<string> feature_arr(batch_size, one_feature);

    timeval tv;
    gettimeofday(&tv, NULL);
    long t1 = tv.tv_sec * 1000 + tv.tv_usec / 1000;

    py::array_t<int> result = deal_batch(feature_arr, m, 100);

    gettimeofday(&tv, NULL);
    long t2 = tv.tv_sec * 1000 + tv.tv_usec / 1000;
    // batch_size = 1000, 耗时在 163
    cout << "batch_size = " << batch_size << " 程序耗时 (ms) :" << t2 - t1 << endl;
    // cout << result << endl;
    return 0;
}

/*
不用 json 序列/反序列化,  batch_size : ms  = 1000 : 98
不用 json 序列/反序列化,  batch_size : ms  = 2000 : 193
*/

/*
cd cpp_ext
c++ -O3 -Wall -shared -std=c++11 -I./include/ $(python3 -m pybind11 --includes) cpp_tokenizer.cc -o cpp_tokenizer.o
*/
