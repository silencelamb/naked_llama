#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//Step 1.  统计
int32_t bincount[256];              // 每个专家的token数量
int32_t export_token_map[256][32];  // 每个专家的token index

for (int32_t i = 0; i < 256; i++) {
    bincount[i] = 0;
}

// 专家索引 expert_indice:  [n_tokens, topk]

for (int32_t token_idx = 0; token_idx < n_tokens*topk; token_idx++) {
    int32_t expert = expert_indice[token_idx];  // 0-255
    // 该专家的token数量加1
    bincount[expert]++;
    // 把该token index添加到该专家的token列表中
    export_token_map[expert][bincount[expert]] = token_idx;
}

//Step 2.  分配每个tile的计算任务
int32_t tile_expert_count[16];         // 每个tile负责计算的专家数量
int32_t tile_expert_idx[16][256];      // 每个tile负责计算的专家index

assign_expert_to_tile(bincount, tile_expert_count, tile_expert_idx);


//Step 3.  每个tile执行自己的计算任务
int32_t TILE_ID = get_tile_id();  // 获取当前tile的ID
// 循环计算当前tile负责的专家
for (int32_t i = 0; i < tile_expert_count[TILE_ID]; i++) {
    int32_t expert_idx = tile_expert_idx[TILE_ID][i];  // 获取当前要计算的专家
    //计算1个专家的Step a: 先gather token输入
    int32_t token_count = bincount[expert_idx];  // 获取该专家的token数量
    for (int32_t j = 0; j < token_count; j++) {
        int32_t token_idx = export_token_map[expert_idx][j];  // 获取该token的index
        // gather token输入
        // 注意这里除以topk 取整，是为了算原始是哪个token
        gather_token_input( int(token_idx/topk) );
    }
    //计算1个专家的Step b: 执行专家计算
    execute_one_expert();

    //计算1个专家的Step c: scatter token输出
    for (int32_t j = 0; j < token_count; j++) {
        int32_t token_idx = export_token_map[expert_idx][j];  // 获取该token的index
        // scatter token输出
        scatter_token_output(token_idx);
    }
}
//Step 4.  乘以weight并reduce结果
// result : [n_tokens*topk, hidden_dim], weight: [n_tokens, topk] -> [n_tokens * topk]

// load -> multiply weight -> reduce 调用合适指令
