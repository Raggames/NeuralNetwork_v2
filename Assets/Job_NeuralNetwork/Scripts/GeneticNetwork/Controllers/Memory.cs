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
        public List<Vector3> Memory_WaterKnownPosition = new List<Vector3>();
        public List<Memory_OtherIndividuals> Memory_OtherIndividuals = new List<Memory_OtherIndividuals>();


        public void ToMemory(Vector3 position, int memPointer)
        {
            if(!TryGetMemoryEntry(memPointer, position)) // Checking memory doesn't contains entry
            {
                //Add memory slot
                if (memPointer == 0) // FOOD
                {
                    Memory_FoodKnownPosition.Add(position);
                }
                else if (memPointer == 1) // WATER
                {
                    Memory_WaterKnownPosition.Add(position);
                }
            }
        }

        public void ToMemory(GeneticInstanceController entity, float interestFactor, int memPointer)
        {
            var tryEntry = TryGetMemoryEntry(memPointer, entity);
            if (tryEntry == null)
            {
                Memory_OtherIndividuals.Add(new Controllers.Memory_OtherIndividuals()
                {
                    Entity = entity,
                    InterestFactor = interestFactor,
                    LastKnownPosition = entity.transform.position,
                });
            }
            else
            {
                // Existing ? Update
                tryEntry.LastKnownPosition = entity.transform.position;
                tryEntry.InterestFactor = interestFactor;
            }
        }

        public Vector3 TryGetMemoryEntry(int memPointer, Vector3 askPosition) // 0 for Food, 1 for Water 2 for Individuals
        {
            if(memPointer == 0) // FOOD
            {
                return Memory_FoodKnownPosition.Find(t => t == askPosition);
            }
            else if(memPointer == 1) // WATER
            {
                return Memory_WaterKnownPosition.Find(t => t == askPosition);
            }
            else
            {
                Debug.LogError("No access in memory at this Pointer");
            }
            return new Vector3(-1,-1,-1);
        }

        public Memory_OtherIndividuals TryGetMemoryEntry(int memPointer, GeneticInstanceController askIndividual) // 0 forIndividuals
        {
            if (memPointer == 0) // FOOD
            {
                var checkEntry = Memory_OtherIndividuals.Find(t => t.Entity == askIndividual);
                if (checkEntry != null)
                {
                    return checkEntry;
                }
            }
            else
            {
                Debug.LogError("No access in memory on this Pointer");
            }
            return null;
        }

        public Memory_OtherIndividuals FromMemory(int memPointer, GeneticInstanceController entity)
        {
            return TryGetMemoryEntry(memPointer, entity);
        }

        public List<Vector3> FromMemory(int memPointer)
        {
            if (memPointer == 0) // FOOD
            {
                if(Memory_FoodKnownPosition.Count > 0)
                {

                }
            }
            else if (memPointer == 1) // WATER
            {
                if(Memory_WaterKnownPosition.Count > 0)
                {

                }
            }
            return null;
        }
    }

    public class Memory_OtherIndividuals
    {
        public float InterestFactor; // Positive if friend, negative if enemy
        public GeneticInstanceController Entity;
        public Vector3 LastKnownPosition;
    }
}
