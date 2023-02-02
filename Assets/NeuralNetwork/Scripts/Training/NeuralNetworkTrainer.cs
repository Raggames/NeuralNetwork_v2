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
        [Header("---- PARAMETERS ----")]
        public Vector2 InitialWeightRange = new Vector2(-.1f, .1f);
        [Range(0.00001f, 2f)] public float LearningRate = .3f;

        public int Epochs;
        public int BatchSize;

        [Header("---- RUNTIME ----")]
        [ReadOnly] public int CurrentEpoch;
    }
}
