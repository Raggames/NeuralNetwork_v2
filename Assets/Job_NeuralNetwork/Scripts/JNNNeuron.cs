using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Unity.Collections;
using UnityEngine;

namespace Assets.Job_NeuralNetwork.Scripts
{
    public class JNNNeuron : MonoBehaviour
    {
        public int Layer;
        public int ID;

        public double Weight;
        public double Bias;
        public double PreviousDelta;

        public double output;
        public double Output
        {
            set
            {
                output = value;
                light.intensity =  10*(float)output;
            }
        }
        public double error;

        public Light light;

        public struct NeuronData
        {
            public int Layer;
            public int ID;

            public double Weight;
            public double Bias;
            public double PreviousDelta;
            public double output;
            public double error;

            public NeuronData(JNNNeuron instance)
            {
                Layer = instance.Layer;
                ID = instance.ID;
                Weight = instance.Weight;
                Bias = instance.Bias;
                PreviousDelta = instance.PreviousDelta;
                output = instance.output;
                error = instance.error;
            }
        }

       

        
    }
}
