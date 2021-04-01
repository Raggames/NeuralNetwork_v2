using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace Assets.Job_NeuralNetwork.Scripts.GeneticNetwork.Controllers
{
    public class Sense : MonoBehaviour
    {
        public GeneticInstanceController geneticController;

        public List<Vector3> TempFoodPosition = new List<Vector3>();
        public List<Vector3> TempWaterPositions = new List<Vector3>();
        public List<GeneticInstanceController> EntitiesAround = new List<GeneticInstanceController>();

        public SphereCollider senseCollider;

        public void Start()
        {
            geneticController = GetComponent<GeneticInstanceController>();
            senseCollider = GetComponent<SphereCollider>();
            senseCollider.radius = geneticController.Traits[1].Value;

        }

        public void OnTriggerEnter(Collider other)
        {
            if (other.CompareTag("Animal"))
            {
                OnDetectEntity(other.GetComponent<GeneticInstanceController>());
            }
            if (other.CompareTag("Food"))
            {
                TempFoodPosition.Add(other.transform.position);
            }
            if (other.CompareTag("Water"))
            {
                TempWaterPositions.Add(other.transform.position);
            }
        }

        public void OnTriggerExit(Collider other)
        {
            if (other.CompareTag("Animal"))
            {
                OnUndetectEntity(other.GetComponent<GeneticInstanceController>());
            }
            if (other.CompareTag("Food"))
            {
                TempFoodPosition.Remove(other.transform.position);
            }
            if (other.CompareTag("Water"))
            {
                TempWaterPositions.Remove(other.transform.position);
            }
        }

        public void OnDetectEntity(GeneticInstanceController controller)
        {
            EntitiesAround.Add(controller);
        }

        public void OnUndetectEntity(GeneticInstanceController controller)
        {
            EntitiesAround.Remove(controller);
        }

        public List<Vector3> GetSensePositionsData(int pointer = 0)
        {
            if(pointer == 0)
            {
                return TempFoodPosition;
            }
            else if (pointer == 1)
            {
                return TempWaterPositions;
            }
            else
            {
                return null;
            }
        }

        public List<GeneticInstanceController> GetSenseEntitiesData()
        {
            return EntitiesAround;
        }

        public void Clean()
        {
            TempFoodPosition.Clear();
            TempWaterPositions.Clear();
            EntitiesAround.Clear();
        }
    }
}
