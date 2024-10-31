using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Unity.Collections;

namespace NeuralNetwork
{
    public static class JobSystemHelpers
    {
        public static NativeArray<double> GetRandomValues(int lenght)
        {
            NativeArray<double> inputs = new NativeArray<double>(lenght, Allocator.Persistent);
            for (int i = 0; i < inputs.Length; ++i)
            {
                inputs[i] = UnityEngine.Random.Range(0f, 10f);
            }
            return inputs;
        }

        public static NativeArray<double> ToNativeArray(double[] arrayIn)
        {
            NativeArray<double> inputs = new NativeArray<double>(arrayIn.Length, Allocator.Persistent);
            for (int i = 0; i < inputs.Length; ++i)
            {
                inputs[i] = arrayIn[i];
            }
            return inputs;
        }

        public static double[] FromNativeArray(NativeArray<double> arrayIn)
        {
            double[] results = new double[arrayIn.Length];
            for (int i = 0; i < results.Length; ++i)
            {
                results[i] = arrayIn[i];
            }
            return results;
        }

    }
}
