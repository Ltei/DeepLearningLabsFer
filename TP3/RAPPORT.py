
# h(t) = tanh(W*h(t-1) + U*x(t) + b)
# o(t) = V * h(t) + c
# y_(t) = softmax(o(t))

# Exercice 2.1

DATA_SIZE_BYTES = 4
HIDDEN_LAYER_LEN = 500
INOUT_LEN = 70 #10000

W_len = HIDDEN_LAYER_LEN * HIDDEN_LAYER_LEN
U_len = INOUT_LEN * HIDDEN_LAYER_LEN
V_len = HIDDEN_LAYER_LEN * INOUT_LEN
b_len = HIDDEN_LAYER_LEN
c_len = INOUT_LEN
total_len = W_len + U_len + V_len + b_len + c_len
total_size_bytes = total_len * DATA_SIZE_BYTES

print(W_len)
print(U_len)
print(V_len)
print(b_len)
print(c_len)
print(total_len)
print(total_size_bytes)
