using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace Assets.Job_NeuralNetwork.Scripts.GeneticNetwork.GeneticInstancesEvaluation
{
    public class Genetic_Animal : GeneticInstanceController
    {
        [Header("Entity DNA Parameters")]
        public int MaxHunger; // Resistance to hunger
        public int MaxLife; // Maximum vitality
        public int Courage; // Resistance to fear, involved in fleeing or attacking decisions
        public int Speed; // MovingSpeed of that entity
        public int Desirability; // Chances of reproduction 

        [Header("Entity RealTime Parameters")]
        public bool IsAlive;

        public int CurrentLife;
        public int CurrentHunger;
        public int CurrentWaterNeed;
        public int CurrentFear;
               
        [Header("Evaluation Parameters")] // Some more meta-parameters to evaluate fitness to an entity in its environnement
        public float SurvivedTime;
        public float FoodEaten; // amount of currentHunger --
        public float NumberOfChilds;


        public override void StartExecution()
        {
            IsAlive = true;
        }

        // NeuralNetwork Computes Controller parameters to output a decision with is controlled here
        #region Decisions
        public override void ExecuteDecision(double[] inputs)
        {
            throw new NotImplementedException();
        }

        #endregion

        #region Actions
        public void SearchForFood() // Could be a prey or vegetables
        {

        }

        public void SearchForWater()
        {

        }

        public void SearchForPartner()
        {

        }

        public void Attack()
        {

        }

        public void Flee()
        {

        }

        public void Wait()
        {

        }
        #endregion

        public override GeneticEvaluationData ComputeEvaluationData()
        {
            GeneticEvaluationData data = new GeneticEvaluationData();
            data.networkInstanceID = geneticBrain.UniqueID;

            double[] evaluate = new double[] // we evaluate fitness on DNA Parameters AND meta-parameters 
            {
                SurvivedTime,
                FoodEaten,
                NumberOfChilds,
                MaxLife,
                MaxHunger,
                Speed,
                Desirability,
                Courage,
            };

            return data;
        }

        public override void Die()
        {
            throw new NotImplementedException();
        }

        // REPRODUCTION : Here is a method to create child instances of two Entities
        public override void Reproduct(GeneticInstanceController partner)
        {
            throw new NotImplementedException();
        }
    }
}
