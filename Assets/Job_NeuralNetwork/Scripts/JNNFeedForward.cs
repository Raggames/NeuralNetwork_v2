using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using static Assets.Job_NeuralNetwork.Scripts.JNNMath;

namespace Assets.Job_NeuralNetwork.Scripts
{
    public class JNNFeedForward : MonoBehaviour
    {
        private JNNManager jnnManager;
        [Header("DONN Architecture")]
        /*public int InputLayerNeurons;
        public List<int> HiddenLayersNeurons;
        public int OutputLayerNeurons;*/

        [SerializeField]
        public JNNFeedForwardLayer InputLayer;
        public List<JNNFeedForwardLayer> HiddenLayers = new List<JNNFeedForwardLayer>();
        public JNNFeedForwardLayer OutputLayer;

        JNNNeuron[] inputLayer;
        List<JNNNeuron[]> hiddenLayers;
        JNNNeuron[] outputLayer;

        public GameObject neuronPrefab;

        private double momentum;
        private double weightDecay;

        [Header("DONN Rendering")]
        public int ScaleXY = 1;
        public int ScaleZ = 1;
      
        // JOB
        public JobHandle handle;

        // Network Running Data
        NativeArray<double> networkInputs;
        NativeArray<double> layerOutputs;

        private double[] networkOutputs;

        // CREATING NETWORK ********************************************************************************************
        public void CreateNetwork(JNNManager manager)
        {
            jnnManager = manager;
            momentum = manager.Momentum;
            weightDecay = manager.WeightDecay;
            // Creating Arrays
            inputLayer = new JNNNeuron[InputLayer.NeuronsCount];

            hiddenLayers = new List<JNNNeuron[]>();
            for (int i = 0; i < HiddenLayers.Count; ++i)
            {
                hiddenLayers.Add(new JNNNeuron[HiddenLayers[i].NeuronsCount]);
            }

            outputLayer = new JNNNeuron[OutputLayer.NeuronsCount];


            // Creating Entities in Array
            Vector3 center = new Vector3();

            int yOffset = 0;
            int indexor = 0;
            int size = (int)math.sqrt((float)inputLayer.Length);

            for (int i = 0; i < inputLayer.Length; ++i)
            {
                if (indexor >= size)
                {
                    yOffset++;
                    indexor = 0;
                }

                GameObject neuron = Instantiate(neuronPrefab, transform);
                JNNNeuron neuronComponent = neuron.GetComponent<JNNNeuron>();
                inputLayer[i] = neuronComponent;

                Vector3 Value = new float3(indexor * ScaleXY, yOffset * ScaleXY, 0f);
                neuronComponent.transform.position = Value;
                center += Value;

                int layer = 1;

                neuronComponent.Layer = layer;
                neuronComponent.ID = SetID(layer, i);
                neuronComponent.Weight = 0;
               
                indexor++;

            }
            center /= inputLayer.Length;

            for (int i = 0; i < inputLayer.Length; ++i)
            {
                inputLayer[i].transform.position -= center;
            }
            


            for (int i = 0; i < hiddenLayers.Count; ++i)
            {
                yOffset = 0;
                indexor = 0;
                size = (int)math.sqrt((float)hiddenLayers[i].Length);

                center = new Vector3();
                for (int j = 0; j < hiddenLayers[i].Length; ++j)
                {
                    if (indexor >= size)
                    {
                        yOffset++;
                        indexor = 0;
                    }
                    GameObject neuron = Instantiate(neuronPrefab, transform);
                    JNNNeuron neuronComponent = neuron.GetComponent<JNNNeuron>();

                    hiddenLayers[i][j] = neuronComponent;

                    Vector3 Value = new float3(indexor * ScaleXY, yOffset * ScaleXY, (i + 1) * ScaleZ);
                    neuronComponent.transform.position = Value;
                    center += Value;

                    int layer = 2 + i;
                    neuronComponent.Layer = layer;
                    neuronComponent.ID = SetID(layer, j);
                    neuronComponent.Weight = GetRandomWeight();
                   

                    indexor++;
                }
                center /= hiddenLayers[i].Length;

                for(int j = 0; j < hiddenLayers[i].Length; ++j)
                {
                    hiddenLayers[i][j].transform.position -= new Vector3(center.x, center.y, 0f);
                }
            }

            yOffset = 0;
            indexor = 0;
            size = (int)math.sqrt((float)outputLayer.Length);

            center = new Vector3();
            for (int i = 0; i < outputLayer.Length; ++i)
            {
                if (indexor >= size)
                {
                    yOffset++;
                    indexor = 0;
                }
                GameObject neuron = Instantiate(neuronPrefab, transform);
                JNNNeuron neuronComponent = neuron.GetComponent<JNNNeuron>();
                outputLayer[i] = neuronComponent;

                Vector3 Value = new float3(indexor * ScaleXY, yOffset * ScaleXY, (hiddenLayers.Count + 1) * ScaleZ);

                neuronComponent.transform.position = Value;
                center += Value;

                int layer = 2 + hiddenLayers.Count;

                neuronComponent.Layer = layer;
                neuronComponent.ID = SetID(layer, i);
                neuronComponent.Weight = GetRandomWeight();
                neuronComponent.Bias = GetRandomWeight();
                
                indexor++;
            }
            center /= outputLayer.Length;
            for (int i = 0; i < outputLayer.Length; ++i)
            {
                outputLayer[i].transform.position -= new Vector3(center.x, center.y, 0f);
            }

        }

        // EXECUTION ***************************************************************************************************
        #region Execution
        public void ComputeFeedForward(NativeArray<double> inputs, out double[] results)
        {
            networkInputs = inputs;
            //layerOutputs = new NativeArray<double>(hiddenLayers[0].Length, Allocator.Persistent);
            ComputeLayer(hiddenLayers[0], networkInputs, out layerOutputs, HiddenLayers[0]);

            for(int i = 1; i < hiddenLayers.Count; ++i)
            {
                ComputeLayer(hiddenLayers[i], layerOutputs, out layerOutputs, HiddenLayers[i]); 
            }

            ComputeLayer(outputLayer, layerOutputs, out layerOutputs, OutputLayer, true);

            results = networkOutputs;

            //layerOutputs.Dispose();
        }

        public void ComputeLayer(JNNNeuron[] layer, NativeArray<double> inputs, out NativeArray<double> outputData, JNNFeedForwardLayer layerData, bool lastLayer = false)
        {
            // On crée des Array temporaires (TempJob) pour stocker les données et les communiquer au Job. Les inputs de la couche précédente, ainsi que la layer à traiter pour récuperer Weight et biais sur chaque neurone
            NativeArray<JNNNeuron.NeuronData> neuronsData = new NativeArray<JNNNeuron.NeuronData>(layer.Length, Allocator.TempJob);
            NativeArray<double> input = inputs;
            NativeArray<double> outputs = new NativeArray<double>(layer.Length, Allocator.TempJob);

            // on utilise le constructeur de JobNeuronCOmponent => NeuronData pour feeder la liste neuronsData
            for (int i = 0; i < layer.Length; ++i)
            {
                neuronsData[i] = new JNNNeuron.NeuronData(layer[i]);

            }

            // On créer le job de traitement de la couche et on y passe sa data.
            switch (layerData.ActivationFunction)
            {
                case ActivationFunctions.Linear:
                    LinearJob linearJob = new LinearJob
                    {
                        dataArray = neuronsData,
                        inputs = inputs,
                        outputs = outputs,
                    };

                    // On Schedule le Job. On peux utiliser => layerJob.Run(neuronsData.Length); pour traiter le Job en mainthread et debugger.
                    handle = linearJob.Schedule(neuronsData.Length, 1);

                    handle.Complete();

                    break;
                case ActivationFunctions.Sigmoid:
                    SigmoidJob sigmoidJob = new SigmoidJob
                    {
                        dataArray = neuronsData,
                        inputs = inputs,
                        outputs = outputs,
                    };

                    // On Schedule le Job. On peux utiliser => layerJob.Run(neuronsData.Length); pour traiter le Job en mainthread et debugger.
                    handle = sigmoidJob.Schedule(neuronsData.Length, 1);

                    handle.Complete();
                    break;

                case ActivationFunctions.Tanh:
                    TanhJob tanhJob = new TanhJob
                    {
                        dataArray = neuronsData,
                        inputs = inputs,
                        outputs = outputs,
                    };

                    // On Schedule le Job. On peux utiliser => layerJob.Run(neuronsData.Length); pour traiter le Job en mainthread et debugger.
                    handle = tanhJob.Schedule(neuronsData.Length, 1);

                    handle.Complete();
                    break;

                case ActivationFunctions.Sinusoid:
                    SinusoidJob sinusoidJob = new SinusoidJob
                    {
                        dataArray = neuronsData,
                        inputs = inputs,
                        outputs = outputs,
                    };

                    // On Schedule le Job. On peux utiliser => layerJob.Run(neuronsData.Length); pour traiter le Job en mainthread et debugger.
                    handle = sinusoidJob.Schedule(neuronsData.Length, 1);

                    handle.Complete();
                    break;

                case ActivationFunctions.ReLU:
                    ReLUJob reLUJob = new ReLUJob
                    {
                        dataArray = neuronsData,
                        inputs = inputs,
                        outputs = outputs,
                    };

                    // On Schedule le Job. On peux utiliser => layerJob.Run(neuronsData.Length); pour traiter le Job en mainthread et debugger.
                    handle = reLUJob.Schedule(neuronsData.Length, 1);

                    handle.Complete();
                    break;
                case ActivationFunctions.PReLU:
                    PReLUJob PreLUJob = new PReLUJob
                    {
                        dataArray = neuronsData,
                        inputs = inputs,
                        outputs = outputs,
                    };

                    // On Schedule le Job. On peux utiliser => layerJob.Run(neuronsData.Length); pour traiter le Job en mainthread et debugger.
                    handle = PreLUJob.Schedule(neuronsData.Length, 1);

                    handle.Complete();
                    break;
                case ActivationFunctions.Softmax:
                    LinearJob beforeSoftMax = new LinearJob
                    {
                        dataArray = neuronsData,
                        inputs = inputs,
                        outputs = outputs,
                    };

                    // On Schedule le Job. On peux utiliser => layerJob.Run(neuronsData.Length); pour traiter le Job en mainthread et debugger.
                    handle = beforeSoftMax.Schedule(neuronsData.Length, 1);

                    handle.Complete();
                                        
                    double[] results = JNNMath.Softmax(outputs.ToArray());
                    for(int i = 0; i < results.Length; ++i)
                    {
                        outputs[i] = results[i];
                    }

                    break;

            }
                                   
            // On sort les données de la couche
            outputData = outputs;

            for(int i = 0; i < layer.Length; ++i)
            {
                if(outputData[i] == 0)
                {
                    Debug.LogError(0);
                }
                layer[i].Output = outputData[i];
            }

            // Utilisation des arrays terminée, on rend la mémoire allouée.
            neuronsData.Dispose();
            inputs.Dispose();

            // Le run est fini, les données ont traversé toutes les couches. On sort les résultats et on Dispose() les outputs.
            if (lastLayer)
            {
                networkOutputs = new double[outputs.Length];

                for (int i = 0; i < outputs.Length; ++i)
                {
                    networkOutputs[i] = outputs[i];
                }

                outputs.Dispose();
            }
        }
        #endregion

        // WEIGHT SETTING **********************************************************************************************
        #region Weights
        public void SetWeights()
        {

        }

        public void GetWeights()
        {

        }
        #endregion

        // BACKPROPAGATION ********************************************************************************************
        #region BackPropagation
        public void BackPropagate(double[] costs, float learningRate)
        {
            // Computing gradient descent
            
            if(OutputLayer.ActivationFunction != ActivationFunctions.Softmax)
            {
                for (int i = 0; i < outputLayer.Length; ++i)
                {
                    outputLayer[i].grad = costs[i] * JNNMath.ComputeActivation(OutputLayer.ActivationFunction, true, outputLayer[i].output);
                }
            }
            else
            {
                double[] alloutputs = new double[outputLayer.Length];
                for (int i = 0; i < outputLayer.Length; ++i)
                {
                    alloutputs[i] = outputLayer[i].output;
                }

                for (int i = 0; i < outputLayer.Length; ++i)
                {
                    outputLayer[i].grad = costs[i] * JNNMath.ComputeActivation(ActivationFunctions.Sigmoid, true, outputLayer[i].output);
                }
            }
           
            for(int i = hiddenLayers.Count -1; i >= 0; --i)
            {
                if(i == hiddenLayers.Count - 1)
                {
                    for (int j = 0; j < hiddenLayers[i].Length; ++j)
                    {
                        double sum = 0f;
                        for (int k = 0; k < outputLayer.Length; ++k)
                        {
                            sum += hiddenLayers[i][j].Weight * outputLayer[k].grad;
                        }

                        hiddenLayers[i][j].grad = sum * JNNMath.ComputeActivation(HiddenLayers[i].ActivationFunction, true, hiddenLayers[i][j].output);
                    }
                }
                else
                {
                    for (int j = 0; j < hiddenLayers[i].Length; ++j)
                    {
                        double sum = 0f;
                        for (int k = 0; k < hiddenLayers[i+1].Length; ++k)
                        {
                            sum += hiddenLayers[i][j].Weight * hiddenLayers[i+1][k].grad;
                        }

                        hiddenLayers[i][j].grad = sum * JNNMath.ComputeActivation(HiddenLayers[i].ActivationFunction, true, hiddenLayers[i][j].output);
                    }
                }
            }
           
            // Now calling weight and bias update on network
            UpdateWeights(learningRate);
        }
        
        public void UpdateWeights(float learningRate)
        {
            // Update Output Bias
            for (int i = 0; i < outputLayer.Length; ++i)
            {
                double delta = learningRate * outputLayer[i].grad;
                outputLayer[i].Bias += delta;
                outputLayer[i].Bias += outputLayer[i].PreviousDelta * momentum;
                outputLayer[i].Bias -= weightDecay * outputLayer[i].Bias;
                outputLayer[i].PreviousDelta = delta;
            }

            // Update Output Weight
            for (int i = 0; i < outputLayer.Length; ++i)
            {
                double delta = learningRate * outputLayer[i].grad * outputLayer[i].Weight;
                outputLayer[i].Weight += delta;
                outputLayer[i].Weight += outputLayer[i].PreviousDelta * momentum;
                outputLayer[i].Weight -= weightDecay * outputLayer[i].Weight;
                outputLayer[i].PreviousDelta = delta;
            }
            

            // Update Hidden Weights
            for (int i = 0; i < hiddenLayers.Count; ++i)
            {
                for (int k = 0; k < hiddenLayers[i].Length; ++k)
                {
                    double delta = learningRate * hiddenLayers[i][k].grad * hiddenLayers[i][k].Weight;
                    hiddenLayers[i][k].Weight += delta;
                    hiddenLayers[i][k].Weight += hiddenLayers[i][k].PreviousDelta * momentum;
                    hiddenLayers[i][k].Weight -= weightDecay * hiddenLayers[i][k].Weight;
                    hiddenLayers[i][k].PreviousDelta = delta;
                }
            }

            // Update Hidden Bias
            for (int i = 0; i < hiddenLayers.Count; ++i)
            {
                for (int k = 0; k < hiddenLayers[i].Length; ++k)
                {
                    double delta = learningRate * hiddenLayers[i][k].grad;
                    hiddenLayers[i][k].Bias += delta;
                    hiddenLayers[i][k].Bias += hiddenLayers[i][k].PreviousDelta * momentum;
                    hiddenLayers[i][k].Bias -= weightDecay * hiddenLayers[i][k].Bias;
                    hiddenLayers[i][k].PreviousDelta = delta;
                }
            }
        }
        #endregion

        // SERIALISATION **********************************************************************************************
        #region Serialisation

        public struct NetworkData
        {
            public int Version;

            public double[] weights;
            public double[] bias;
            public double[] momentum;

        }

        public string saveName;


        public string CreateSaveName(int version)
        {
            string newName = "";

            saveName = newName;
            return saveName;
        }

        private void JNN_Save(NetworkData data)
        {
          
            // Serialize
            JNNSerializer.Save(data, CreateSaveName(data.Version));
        }

        private NetworkData JNN_Load(string fileName)
        {
            NetworkData loadedData = new NetworkData();
            loadedData = JNNSerializer.Load(loadedData, fileName);
            return loadedData;
        }

        #endregion

        // UTILS ******************************************************************************************************
        #region Utils
        public NativeArray<double> GetRandomValues()
        {
            NativeArray<double> inputs = new NativeArray<double>(InputLayer.NeuronsCount, Allocator.Persistent);
            for (int i = 0; i < inputs.Length; ++i)
            {
                inputs[i] = UnityEngine.Random.Range(0f, 10f);
            }
            return inputs;
        }

        public NativeArray<double> ToNativeArray(double[] arrayIn)
        {
            NativeArray<double> inputs = new NativeArray<double>(InputLayer.NeuronsCount, Allocator.Persistent);
            for (int i = 0; i < inputs.Length; ++i)
            {
                inputs[i] = arrayIn[i];
            }
            return inputs;
        }

        public double GetRandomWeight()
        {
            return UnityEngine.Random.Range(-1f, 1f);
        }


        public static int SetID(int Layer, int index)
        {
            string ID = Layer.ToString() + index.ToString();
            int iD = int.Parse(ID);
            return iD;
        }
        #endregion
    }
}
