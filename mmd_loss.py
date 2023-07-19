import tensorflow as tf

# Maximum mean discrepancy calculation
def mix_rbf_kernel(X, Y, sigmas, wts=None):
    if wts is None:
        wts = [1] * len(sigmas)

    XX = tf.matmul(X, X, transpose_b=True)
    XY = tf.matmul(X, Y, transpose_b=True)
    YY = tf.matmul(Y, Y, transpose_b=True)

    X_sqnorms = tf.linalg.diag_part(XX)
    Y_sqnorms = tf.linalg.diag_part(YY)

    r = lambda x: tf.expand_dims(x, 0)
    c = lambda x: tf.expand_dims(x, 1)

    K_XX, K_XY, K_YY = 0, 0, 0
    for sigma, wt in zip(sigmas, wts):
        gamma = 1 / (2 * sigma**2)
        K_XX += wt * tf.exp(-gamma * (-2 * XX + c(X_sqnorms) + r(X_sqnorms)))
        K_XY += wt * tf.exp(-gamma * (-2 * XY + c(X_sqnorms) + r(Y_sqnorms)))
        K_YY += wt * tf.exp(-gamma * (-2 * YY + c(Y_sqnorms) + r(Y_sqnorms)))

    return K_XX, K_XY, K_YY, tf.reduce_sum(wts)


def mix_rbf_mmd2(X, Y, sigmas=(1,), wts=None):
    K_XX, K_XY, K_YY, d = mix_rbf_kernel(X, Y, sigmas, wts)
    m = tf.cast(tf.shape(K_XX)[0], tf.float32)
    n = tf.cast(tf.shape(K_YY)[0], tf.float32)

    mmd2 = (tf.reduce_sum(K_XX) / (m * m) + tf.reduce_sum(K_YY) / (n * n) - 2 * tf.reduce_sum(K_XY) / (m * n))
    return mmd2
