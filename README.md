# Tumor-As-Quantum-Particle
This is the implementation for article entitled: "Treatment-Response Analysis of Tumor as A Quantum Particle".

## Introduction
Quantum computing is a new technology that promises a new way to accelerate or perhaps revise our look at the current Machine Learning (ML) models. It is noted on the current proposal for Geometric Deep Learning that any ML model is a group action on set\cite{bronstein2017geometric,bronstein2021geometric}; here, the set includes input signal, and the group of action can be considered as functor. Current Euclidean embedding is representation in $GL_{n\times m}(\mathbb{R})$, while quantum machine learning models learn the embeddings on $GL_{n\times m}(\mathbb{C})$. Our previous works have addressed several applications\cite{nguyen2022quantum,nguyen2022bayesian} of quantum neural networks, which show agreement with the early literature of the field on the potential of using quantum computing to unlock the full potential of Artificial Intelligence.


## Quantum Neural Networks vs. Classical Neural Networks
A classical neural network is given as a parameterized function
$$\hat{y} = f_{\bm{\theta}}(X, y);$$ where $X$ is the training input, $y$ is the label and $\hat{y}$ is the predicted values. The transformation of data in the classical model is presented as 
\begin{equation}
	\begin{split}
	f: GL(\mathbb{R}^p) \rightarrow GL(\mathbb{R}^q)\\
	X_{p=n \times m} \rightarrow X'_{q = n' \times m'}
	\end{split}
\end{equation}

Quantum neural networks instead using transformation on the Hilbert vector space $\mathcal{H}$ using these following rotation in Ox, Oy, Oz axis parameterized by $\theta_{ij}$ given by $$R_{\sigma_x}(\bm{\theta}) =  \begin{bmatrix}
\cos (\bm{\theta}/2) & -i \sin(\bm{\theta}/2)\\
-i \sin(\bm{\theta}/2) & \cos (\bm{\theta}/2)
\end{bmatrix},$$ $$R_{\sigma_y}(\bm{\theta}) =  \begin{bmatrix}
\cos (\bm{\theta}/2) & -\sin(\bm{\theta}/2)\\
\sin(\bm{\theta}/2) & \cos (\bm{\theta}/2)
\end{bmatrix},$$ and $$R_{\sigma_z}(\bm{\theta}) = \begin{bmatrix}
e^{-i\frac{\bm{\theta}}{2}} & 0\\
0 & e^{i\frac{\bm{\theta}}{2}}
\end{bmatrix}.$$
We give some comparisons between QNNs to some model architecture:
\begin{enumerate}
    \item QNNs can be considered as a capsule-neural network\cite{sabour2017dynamic, hinton2018matrix} if we consider the representation before the measurement of the quantum electronic wavefunction. The quantum neuron ("quron") encodes data by a complex-valued functor.
    \item Transformation inside quantum neural network has the geometric information of functor used, represented via Pauli-based (3D) transformations.
    \item QNNs can be viewed as physical-based machine learning because their representations are presented as electronic wavefunctions.
\end{enumerate}


## Contribution
In this work, we aim to achieve the following goals:
\begin{enumerate}
    \item Translation of the theoretical model in \textbf{Equation}~\ref{equa:survival} for learning the non-linear dynamics of tumor evolution, discussed in \textbf{Section}~\ref{section:non_linear_tumor}.
    \item A loss module will be introduced in \textbf{Section}~\ref{sec:loss_module} for efficient training of the quantum model in the context of PFS prediction.
    \item We propose three ways to explain the model prediction. Of note, explainable AI is a hot topic in the current ML literature, as we also attempt to improve the model explainability in this work. Our model prediction delivers three main analyses:
    \begin{enumerate}
        \item Global observations of the entire cohort, including (1) the progression-free probability and (2) response score.
        \item Sub-class specific prediction - T.A.R.G.E.T plots, which quantizes prediction surfaces into different patient classes.
    \end{enumerate}
\end{enumerate}
