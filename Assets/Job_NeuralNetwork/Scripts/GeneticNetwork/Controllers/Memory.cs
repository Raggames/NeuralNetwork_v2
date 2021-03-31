using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace Assets.Job_NeuralNetwork.Scripts.GeneticNetwork.Controllers
{
    public class Memory : MonoBehaviour
    {
        [Header("Memories Lists")]
        public List<Vector3> Memory_FoodKnownPosition = new List<Vector3>();
        public List<Vector3> Memory_KnownWaterPosition = new List<Vector3>();
        public List<Memory_OtherIndividuals> Memory_OtherIndividuals = new List<Memory_OtherIndividuals>();


        public void ToMemory(Vector3 position)
        {

        }

        public void ToMemory(Memory_OtherIndividuals entityData)
        {

        }

        public bool CheckMemory(int memPointer, Vector3 askPosition) // 0 for Food, 1 for Water 2 for Individuals
        {
            if(memPointer == 0) // FOOD
            {
                
            }
            else if(memPointer == 1) // WATER
            {

            }
            else
            {
                Debug.LogError("No access in memory on this Pointer");
            }
            return false;
        }

        public bool CheckMemory(int memPointer, GeneticInstanceController askIndividual) // 0 forIndividuals
        {
            if (memPointer == 0) // FOOD
            {
                if(Memory_OtherIndividuals.Exists(t => t.Entity == askIndividual))
                {
                    return true;
                }
            }
            else
            {
                Debug.LogError("No access in memory on this Pointer");
            }
            return false;
        }

        public Memory_OtherIndividuals FromMemory(GeneticInstanceController entity)
        {
            return new Memory_OtherIndividuals();
        }
        public Vector3 FromMemory(int index = 0)
        {
            return new Vector3();
        }
    }

    public struct Memory_OtherIndividuals
    {
        public float InterestFactor; // Positive if friend, negative if enemy
        public GeneticInstanceController Entity;
        public float LastKnownPosition;
    }
}
