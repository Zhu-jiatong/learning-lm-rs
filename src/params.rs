use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

// implementation inspired by @GYHPCG
impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        let get_tensor = |name: &str| -> Tensor<f32> {
            let tensor = safetensor.tensor(name).unwrap();
            let data = tensor.data();
            let shape = tensor.shape();

            let f32_data = data
                .chunks(4)
                .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
                .collect::<Vec<f32>>();

            return Tensor::new(f32_data, &shape.to_vec());
        };

        let get_layer_tensors = |suffix: &str, layers_count: usize| -> Vec<Tensor<f32>> {
            return (0..layers_count)
                .map(|i| get_tensor(&format!("model.layers.{}.{}", i, suffix)))
                .collect();
        };

        let num_layers = config.num_hidden_layers;
        LLamaParams {
            embedding_table: get_tensor("lm_head.weight"),
            rms_att_w: get_layer_tensors("input_layernorm.weight", num_layers),
            wq: get_layer_tensors("self_attn.q_proj.weight", num_layers),
            wk: get_layer_tensors("self_attn.k_proj.weight", num_layers),
            wv: get_layer_tensors("self_attn.v_proj.weight", num_layers),
            wo: get_layer_tensors("self_attn.o_proj.weight", num_layers),
            rms_ffn_w: get_layer_tensors("post_attention_layernorm.weight", num_layers),
            w_up: get_layer_tensors("mlp.up_proj.weight", num_layers),
            w_gate: get_layer_tensors("mlp.gate_proj.weight", num_layers),
            w_down: get_layer_tensors("mlp.down_proj.weight", num_layers),
            rms_out_w: get_tensor("model.norm.weight"),
            lm_head: get_tensor("lm_head.weight"),
        }
    }
}
