# Transfer Learning with uncertain features

The thesis I am working on as the final step of my bachelors at
Rijksuniversiteit Groningen is an exploratory research in the field of
Uncertainty Quantification which represents a niche of research in
Deep Learning. Its research spans from the need for reliability and
confidence in critical applications of Deep Neural Networks where
overconfidence is detrimental for performance or even dangerous. The
key inisght of UQ is to employ probability theory in order to model
the uncertainty in both the model and the data in order to allow the
NN to back its predictions with confidence scores or reject to make a
prediction if its confidence is too low. In my thesis I will explore
whether Uncertain Features (the byproduct of intermediate layers of a
Bayesian Neural Network having mean and variance) are more informative
than Point Features (the byproduct of intermediate layers of a
Convolutional Neural Networks which is not a probability
distribution). In practice this comparison will be carried out by
using both a BNN and a CNN as Feature Extractors in a Transfer
Learning task and assess which architecture leads to better results in
learning the target data.