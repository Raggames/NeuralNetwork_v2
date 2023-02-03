using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Unity.Collections;
using UnityEngine;

namespace NeuralNetwork
{
    public abstract class NeuralNetworkTrainer : MonoBehaviour
    {
        [Header("---- TRAINING SETTING")]
        /// <summary>
        /// Contains the data set and the function to evaluate the accuracy of the network while training
        /// </summary>
        public TrainingSettingBase TrainingSetting;

        [Header("---- PARAMETERS ----")]
        /// <summary>
        /// The range of the weigths at initialization
        /// </summary>
        public Vector2 InitialWeightRange = new Vector2(-.1f, .1f);
        /// <summary>
        /// The random seed for initializing weigths
        /// </summary>
        public int InitialWeightSeed = 0;
        /// <summary>
        /// The multiplier for tweaking weights each epoch/batch
        /// </summary>
        [Range(0.00001f, 2f)] public float LearningRate = .3f;

        /// <summary>
        /// Momentum is a ratio that allows to add to the new computed weights a part of the vector of the previous iteration.
        /// It allows the learning to be faster by decreasing the fluctuations of the process.
        /// </summary>
        public float Momentum = 0.01f;

        /// <summary>
        /// The learning rate for the biases
        /// </summary>
        public float BiasRate = 1;

        /// <summary>
        /// Control the decreasing of the learning rate over epochs
        /// </summary>
        public float LearningRateDecay = 0.0001f;

        /// <summary>
        /// Decreases all the weigths of the network over epochs (should be a very small amount, it can totally cancel the learning if to close to the learning rate.
        /// </summary>
        public float WeightDecay = 0.0001f;

        /// <summary>
        /// Number of iterations of the learning process
        /// </summary>
        public int Epochs;
        /// <summary>
        /// [Not yet implemented] Batch are a way to accumulate iterations before updating weights (accumulating gradients or error)
        /// </summary>
        public int BatchSize;

        [Header("---- RUNTIME ----")]
        [ReadOnly] public int CurrentEpoch;
    }
}
