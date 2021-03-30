from sklearn.utils.extmath import softmax
campaign_idx = range(0, n_campaigns)
keras_logreg = model.get_layer('contributions').get_weights()[0].flatten()[0:n_campaigns]
keras_logreg = softmax([keras_logreg]).flatten()