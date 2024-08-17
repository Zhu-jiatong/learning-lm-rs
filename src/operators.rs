use crate::tensor::Tensor;

// get (row) vectors from a 2D table given a list of indices
pub fn gather(y: &mut Tensor<f32>, indices: &Tensor<u32>, table: &Tensor<f32>) {
    let length = indices.size();
    let table_shape = table.shape();
    assert!(table_shape.len() == 2);
    let dim = table_shape[1];
    assert!(y.size() == length * dim);
    for i in 0..length {
        let src = &table.data()[indices.data()[i] as usize * dim..][..dim];
        let dst = &mut unsafe { y.data_mut() }[i * dim..][..dim];
        dst.copy_from_slice(src);
    }
}

// RoPE: Rotary Positional Embedding
pub fn rope(y: &mut Tensor<f32>, start_pos: usize, theta: f32) {
    let shape = y.shape();
    assert!(shape.len() == 3);
    let seq_len = shape[0];
    let n_heads = shape[1];
    let d = shape[2];
    let data = unsafe { y.data_mut() };
    for tok in 0..seq_len {
        let pos = start_pos + tok;
        for head in 0..n_heads {
            for i in 0..d / 2 {
                let a = data[tok * n_heads * d + head * d + i];
                let b = data[tok * n_heads * d + head * d + i + d / 2];
                let freq = pos as f32 / theta.powf((i * 2) as f32 / d as f32);
                let (sin, cos) = freq.sin_cos();
                data[tok * n_heads * d + head * d + i] = a * cos - b * sin;
                data[tok * n_heads * d + head * d + i + d / 2] = b * cos + a * sin;
            }
        }
    }
}

// softmax(x) = exp(x - max) / sum(exp(x - max))
// y = softmax(mask(x))
pub fn masked_softmax(y: &mut Tensor<f32>) {
    let ndim = y.shape().len();
    assert!(ndim >= 2);
    let seq_len = y.shape()[ndim - 2];
    let total_seq_len = y.shape()[ndim - 1];
    let batch = y.size() / (seq_len * total_seq_len);
    let data = unsafe { y.data_mut() };
    for b in 0..batch {
        let base = b * seq_len * total_seq_len;
        for i in 0..seq_len {
            let offset = base + i * total_seq_len;
            let boundary = total_seq_len - seq_len + i + 1;

            let max = data[offset..offset + boundary]
                .iter()
                .fold(data[offset], |a, b| a.max(*b));

            let sum = (0..boundary)
                .map(|j| {
                    let e = (data[offset + j] - max).exp();
                    data[offset + j] = e;
                    e
                })
                .sum::<f32>();

            (0..boundary).for_each(|j| data[offset + j] /= sum);
            (boundary..total_seq_len).for_each(|j| data[offset + j] = 0.0);
        }
    }
}

pub fn rms_norm(y: &mut Tensor<f32>, x: &Tensor<f32>, w: &Tensor<f32>, epsilon: f32) {
    let shape = y.shape();
    let (_i, _j) = (shape[0], shape[1]);
    assert!(shape.len() == 2);
    assert!(x.shape() == shape);
    assert!(w.shape().len() == 1);
    assert!(w.size() == _j);

    let _y = unsafe { y.data_mut() };
    let _x = x.data();
    let _w = w.data();

    fn rms(x_i: &[f32], w: &[f32], epsilon: f32) -> Vec<f32> {
        let sum = x_i.iter().fold(0.0, |acc, &x| acc + x * x);
        let n = x_i.len() as f32;
        let deno = (sum / n + epsilon).sqrt();
        let ret = x_i
            .iter()
            .zip(w.iter())
            .map(|(&x, &w)| x * w / deno)
            .collect::<Vec<_>>();
        return ret;
    }

    for i in 0.._i {
        let x_i = &_x[i * _j..][.._j];
        let y_i = rms(x_i, _w, epsilon);
        _y[i * _j..][.._j].copy_from_slice(&y_i);
    }
}

// y = sigmoid(x) * x * y
// hint: this is an element-wise operation
pub fn silu(y: &mut Tensor<f32>, x: &Tensor<f32>) {
    let len = y.size();
    assert!(len == x.size());

    let _y = unsafe { y.data_mut() };
    let _x = x.data();

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    let iter = _y.iter_mut().zip(_x.iter());
    for (y, &x) in iter {
        *y *= sigmoid(x) * x;
    }
}

// C = beta * C + alpha * A @ B^T
// hint: You don't need to do an explicit transpose of B
pub fn matmul_transb(c: &mut Tensor<f32>, beta: f32, a: &Tensor<f32>, b: &Tensor<f32>, alpha: f32) {
    fn transpose(a: &Tensor<f32>) -> Tensor<f32> {
        let shape = a.shape();
        let (m, n) = (shape[0], shape[1]);
        let data = a.data();
        let mut ret = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                ret[j * m + i] = data[i * n + j];
            }
        }
        return Tensor::new(ret, &vec![n, m]);
    }

    fn matmul(a: &Tensor<f32>, b: &Tensor<f32>) -> Tensor<f32> {
        fn dot(x: &[f32], y: &[f32]) -> f32 {
            return x.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum();
        }

        let shape_a = a.shape();
        let shape_b = b.shape();
        let (m_a, n_a) = (shape_a[0], shape_a[1]);
        let (m_b, n_b) = (shape_b[0], shape_b[1]);
        assert!(n_a == m_b);
        let data_a = a.data();
        let data_b = b.data();
        let mut ret = vec![0.0; m_a * n_b];

        for i in 0..m_a {
            for j in 0..n_b {
                let row = &data_a[i * n_a..][..n_a];
                let col = &data_b[j..].iter().step_by(n_b).cloned().collect::<Vec<_>>();
                ret[i * n_b + j] = dot(row, col);
            }
        }

        return Tensor::new(ret, &vec![m_a, n_b]);
    }

    fn scaler_mul_mat(c: &Tensor<f32>, scaler: f32) -> Tensor<f32> {
        let data = c.data();
        let ret = data.iter().map(|&x| x * scaler).collect::<Vec<_>>();
        return Tensor::new(ret, &c.shape());
    }

    let B_T = transpose(b);
    let A_B_T = matmul(a, &B_T);
    let alpha_A_B_T = scaler_mul_mat(&A_B_T, alpha);
    let beta_C = scaler_mul_mat(c, beta);
    let result = matadd(&beta_C, &alpha_A_B_T);
    *c = result;
}

pub fn matadd(a: &Tensor<f32>, b: &Tensor<f32>) -> Tensor<f32> {
    let shape = a.shape();
    let data_a = a.data();
    let data_b = b.data();
    let ret = data_a
        .iter()
        .zip(data_b.iter())
        .map(|(&a, &b)| a + b)
        .collect::<Vec<_>>();

    return Tensor::new(ret, &shape);
}

// Dot product of two tensors (treated as vectors)
#[allow(unused)]
pub fn dot(x: &Tensor<f32>, y: &Tensor<f32>) -> f32 {
    let len = x.size();
    assert!(len == y.size());
    let x_ = x.data();
    let y_ = y.data();
    let mut sum = 0.0;
    for i in 0..len {
        sum += x_[i] * y_[i];
    }
    sum
}

// Sample a index from a tensor (treated as a probability vector)
pub fn random_sample(x: &Tensor<f32>, top_p: f32, top_k: u32, temperature: f32) -> u32 {
    assert!(x.shape()[x.shape().len() - 1] == x.size());
    if temperature <= 0. || top_k < 2 || top_p <= 0. {
        return x
            .data()
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0 as _;
    }

    #[derive(Clone, Copy, PartialEq, Debug)]
    struct Probability {
        val: f32,
        tok: u32,
    }
    impl Eq for Probability {}
    impl PartialOrd for Probability {
        #[inline]
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Probability {
        #[inline]
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            match self.val.total_cmp(&other.val) {
                std::cmp::Ordering::Equal => self.tok.cmp(&other.tok),
                ord => ord.reverse(),
            }
        }
    }
    impl From<(usize, &f32)> for Probability {
        #[inline]
        fn from((i, p): (usize, &f32)) -> Self {
            Self {
                val: p.clone(),
                tok: i as _,
            }
        }
    }

    // sort
    let mut logits = x
        .data()
        .iter()
        .enumerate()
        .map(Probability::from)
        .collect::<Vec<_>>();
    logits.sort_unstable();
    let max = core::mem::replace(&mut logits[0].val, 1.);
    // softmax & sum
    for i in 1..logits.len() {
        logits[i].val = logits[i - 1].val + ((logits[i].val - max) / temperature).exp();
    }
    // topk & topp & random
    let pk = logits[(top_k as usize).min(logits.len()) - 1].val;
    let pp = logits[logits.len() - 1].val * top_p;
    let plimit = rand::random::<f32>() * f32::min(pk, pp);
    // sample
    logits.iter().find(|p| p.val >= plimit).unwrap().tok
}

// Your implementation should at least pass the following tests:
#[test]
fn test_silu() {
    let mut y = Tensor::<f32>::new(vec![2., 3., 4.], &vec![1, 3]);
    let x = Tensor::<f32>::new(vec![1., 2., 3.], &vec![1, 3]);
    silu(&mut y, &x);
    assert!(y.close_to(
        &Tensor::<f32>::new(vec![1.4621172, 5.2847824, 11.43089], &vec![1, 3]),
        1e-3
    ));
}

#[test]
fn test_rms_norm() {
    let mut y = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let x = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let w = Tensor::<f32>::new(vec![1., 2.], &vec![2]);
    rms_norm(&mut y, &x, &w, 1e-6);
    assert!(y.close_to(
        &Tensor::<f32>::new(
            vec![0.6324554, 2.5298216, 0.8485281, 2.2627416],
            &vec![2, 2]
        ),
        1e-3
    ));
}

#[test]
fn test_matmul_transb() {
    let mut c = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let a = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    let b = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    matmul_transb(&mut c, 1., &a, &b, 1.);
    assert!(c.close_to(
        &Tensor::<f32>::new(vec![15., 34., 35., 81.], &vec![2, 2]),
        1e-3
    ));
}
