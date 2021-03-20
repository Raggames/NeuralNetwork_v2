/*using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Entities;
using Unity.Jobs;
using Assets.Scripts;
using Unity.Transforms;
using Unity.Rendering;
using Unity.Collections;
using Unity.Mathematics;

public class ECSNeuralNetwork : MonoBehaviour
{
    public static ECSNeuralNetwork instance;

    EntityManager entityManager;
    EntityArchetype neuronArchetype;

    [Header("DONN Architecture")]
    public int InputLayerNeurons;
    public List<int> HiddenLayersNeurons;
    public int OutputLayerNeurons;

    NativeArray<Entity> inputLayer;
    List<NativeArray<Entity>> hiddenLayers;
    NativeArray<Entity> outputLayer;

    public enum ActivationFunctions
    {
        Linear,
        Sigmoid,
        Boolean,
        Softmax,
    }
    public ActivationFunctions activationFunctions;

    [Header("DONN Rendering")]
    public int Scale = 1;
    public Mesh mesh;
    public Material material;

    // JOB

    JobHandle handle;

    public NativeArray<double> GetInputs()
    {
        return new NativeArray<double>();
    }


    // Start is called before the first frame update
    void Start()
    {
        if(instance == null)
        {
            instance = this;
        }
        else if(instance != null && instance != this)
        {
            Destroy(this);
        }

        entityManager = World.DefaultGameObjectInjectionWorld.EntityManager;

        // Creating Neuron Archetype

        neuronArchetype = entityManager.CreateArchetype(
            typeof(ECSNeuronComponent),
            typeof(Translation),
            typeof(RenderBounds),
            typeof(LocalToWorld),
            typeof(RenderMesh)) ;

        // Creating Arrays
        inputLayer = new NativeArray<Entity>(InputLayerNeurons, Allocator.Persistent);

        hiddenLayers = new List<NativeArray<Entity>>();
        for (int i = 0; i < HiddenLayersNeurons.Count; ++i)
        {
            hiddenLayers.Add(new NativeArray<Entity>(HiddenLayersNeurons[i], Allocator.Persistent));
        }

        outputLayer = new NativeArray<Entity>(OutputLayerNeurons, Allocator.Persistent);


        // Creating Entities in Array

        entityManager.CreateEntity(neuronArchetype, inputLayer);

        int yOffset = 0;
        int indexor = 0;
        int size = (int)math.sqrt((float)inputLayer.Length);
         
        for (int i = 0; i < inputLayer.Length; ++i)
        {
            if(indexor >= size)
            {
                yOffset++;
                indexor = 0;
            }
            entityManager.SetComponentData(inputLayer[i], new Translation 
            {
                Value = new float3(indexor*Scale, yOffset*Scale, 0f)
            });

            entityManager.SetSharedComponentData(inputLayer[i], new RenderMesh
            {
                mesh = mesh,
                material = material,
            });

            int layer = 1;
            entityManager.SetComponentData(inputLayer[i], new ECSNeuronComponent 
            {
                Layer = layer,
                ID = SetID(layer, i),
                Weight = 0.5f
            });

            indexor++;

        }
        inputLayer.Dispose();

        
        for (int i = 0; i < hiddenLayers.Count; ++i)
        {
            entityManager.CreateEntity(neuronArchetype, hiddenLayers[i]);

            yOffset = 0;
            indexor = 0;
            size = (int)math.sqrt((float)hiddenLayers[i].Length);

            for (int j = 0; j < hiddenLayers[i].Length; ++j)
            {
                if (indexor >= size)
                {
                    yOffset++;
                    indexor = 0;
                }

                entityManager.SetComponentData(hiddenLayers[i][j], new Translation 
                { 
                    Value = new float3(indexor*Scale, yOffset*Scale, (i + 1)*Scale)
                });

                entityManager.SetSharedComponentData(hiddenLayers[i][j], new RenderMesh
                {
                    mesh = mesh,
                    material = material,
                });

                int layer = 2 + i;
                entityManager.SetComponentData(hiddenLayers[i][j], new ECSNeuronComponent 
                {
                    Layer = layer,
                    ID = SetID(layer, j),
                    Weight = 0.5f
                });

                indexor++;
            }
            hiddenLayers[i].Dispose();

        }


        entityManager.CreateEntity(neuronArchetype, outputLayer);

        yOffset = 0;
        indexor = 0;
        size = (int)math.sqrt((float)outputLayer.Length);

        for (int i = 0; i < outputLayer.Length; ++i)
        {
            if (indexor >= size)
            {
                yOffset++;
                indexor = 0;
            }

            entityManager.SetComponentData(outputLayer[i], new Translation
            {
                Value = new float3(indexor*Scale, yOffset*Scale, (hiddenLayers.Count + 1)*Scale)
            });

            entityManager.SetSharedComponentData(outputLayer[i], new RenderMesh
            {
                mesh = mesh,
                material = material,
            });

            int layer = 2 + hiddenLayers.Count;
            entityManager.SetComponentData(outputLayer[i], new ECSNeuronComponent
            { 
                Layer = layer,
                ID = SetID(layer, i),
                Weight = 0.5f
            });

            indexor++;
        }
        outputLayer.Dispose();
        
    }

    
    public static int SetID(int Layer, int index)
    {
        string ID = Layer.ToString() + index.ToString();
        int iD = int.Parse(ID);
        return iD;
    }

}
*/