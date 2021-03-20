using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Unity.Collections;
using Unity.Entities;
using Unity.Jobs;

namespace Assets.Scripts
{
    /*public class NeuralNetworkJobSystem : JobComponentSystem
    {
      
        public struct ExecuteLayerJob : IJobForEachWithEntity<NeuronComponent>
        {
            public int Layer;

            public NativeArray<double> inputs;
            public NativeArray<double> outputs;

            [DeallocateOnJobCompletion] public NativeArray<Entity> Neurons;
            
            public void Execute(Entity entity, int index, ref NeuronComponent neuronComponent)
            {
                if(neuronComponent.Layer == Layer)
                {
                    for (int i = 0; i < neuronComponent.Inputs.Length; ++i)
                    {
                        outputs[0] += neuronComponent.Inputs[i] * neuronComponent.Weight;
                    }
                    outputs[0] += neuronComponent.Bias;
                    outputs[0] /= neuronComponent.Inputs.Length;

                    // TODO Activation function

                    neuronComponent.Output = outputs[0];
                }
            }
        }

        protected override JobHandle OnUpdate(JobHandle inputDeps)
        {
            int layerIndex = 0;

            NativeArray<double> output = new NativeArray<double>(1, Allocator.TempJob);
            var executeLayerJob = new ExecuteLayerJob
            {
                Layer = layerIndex,
                inputs = NeuralNetwork.instance.GetInputs(),
                outputs = output, 
            };
            JobHandle handle = executeLayerJob.Schedule(this, inputDeps);
            handle.Complete();
            output.Dispose();

            layerIndex++;
            var executeLayerJob2 = new ExecuteLayerJob
            {
                Layer = layerIndex,
                inputs = NeuralNetwork.instance.GetInputs(),
            };
            executeLayerJob2.Schedule(this, inputDeps);


            return inputDeps;
        }


    }*/
}
