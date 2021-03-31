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

        public List<Vector3> FoodPositions = new List<Vector3>();
        public List<Vector3> WaterPositions = new List<Vector3>();
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
            
        }

        public void OnTriggerExit(Collider other)
        {
            
        }

        public void Refresh()
        {

        }
    }
}
