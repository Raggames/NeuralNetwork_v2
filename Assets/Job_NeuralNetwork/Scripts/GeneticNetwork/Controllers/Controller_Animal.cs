using Assets.Job_NeuralNetwork.Scripts.GeneticNetwork.Controllers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace Assets.Job_NeuralNetwork.Scripts.GeneticNetwork.GeneticInstancesEvaluation
{
    public class Controller_Animal : GeneticInstanceController
    {
        private Memory_Animal memory;
        [Header("Entity DNA Traits")]
        public GenderType Gender;
        public enum GenderType
        {
            Female,
            Male,
        }

        [Header("Entity RealTime Parameters")]
        public bool IsAlive;

        public float CurrentLife;
        public float CurrentHunger;
        public float CurrentWaterNeed;
        public float CurrentFear;
        public float CurrentReproductionNeed;

        public float TimeAtBorn;

        [Header("Brain Executions")]
        protected double[] inputs;
        protected double[] outputs;


        private List<Vector3> foodPosTemp = new List<Vector3>();
        private List<Vector3> waterPosTemp = new List<Vector3>();
        private List<GeneticInstanceController> entitiesTemp = new List<GeneticInstanceController>();

        // ************************************************************************************************************
        #region Initialisation
        public override void Init(GeneticEvolutionManager EvolutionManager, List<Gene> DnaTraits, double[] neuralDna)
        {
            base.Init(EvolutionManager, DnaTraits, neuralDna);
            memory = GetComponent<Memory_Animal>();

            RandGender();
            inputs = new double[geneticBrain.FFNetwork.InputLayer.NeuronsCount];

            if (DnaTraits != null)
            {
                Traits = DnaTraits;
            }
            else
            {
                // Randomizing by a delta on all traits value to get some chaos in individutes at start
                for (int i = 0; i < Traits.Count; ++i)
                {
                    Traits[i] = new Gene(0, Traits[i].TraitName, RandomizeByDelta(Traits[i].Value, 0.1f), RandomizeByDelta(Traits[i].Dominance, 0.05f));
                }
            }

            if(neuralDna != null)
            {
                geneticBrain.FFNetwork.SetWeights(neuralDna);
            }
        }

        public override void Born()
        {
            IsAlive = true;
            CurrentLife = Traits[3].Value;

            TimeAtBorn = Time.realtimeSinceStartup;
        }

        private void RandGender()
        {
            float rand = UnityEngine.Random.Range(0f, 1f);
            if (rand > 0.5f)
            {
                Gender = GenderType.Female;
            }
            else
            {
                Gender = GenderType.Male;
            }
        }

        private float RandomizeByDelta(float input, float delta)
        {
            float rand = UnityEngine.Random.Range(-delta, delta);
            return input + rand;
        }
        #endregion

        // PERCEPTION *************************************************************************************************
        #region Perception
        public override void GetSenseRefresh()
        {
           foodPosTemp = sense.GetSensePositionsData(0);
           waterPosTemp = sense.GetSensePositionsData(1);
           entitiesTemp = sense.GetSenseEntitiesData();
        }

        public float EvaluateFear()
        {
            float maxFear = 0;
            float fear = 0;
            float dist = 0;

            for(int i = 0; i < entitiesTemp.Count; ++i)
            {
                dist = (entitiesTemp[i].transform.position - transform.position).magnitude;
                var mem = memory.TryGetMemoryEntry(0, entitiesTemp[i]);
                float factor = 0;
                if(mem != null)
                {

                    factor = mem.InterestFactor;
                }
                fear = ((Traits[1].Value - dist) / Traits[1].Value) - factor;
                if(fear > maxFear)
                {
                    maxFear = fear;
                }
            }
            return maxFear;
        }

        public float EvaluateNeed(List<Vector3> needInSight)
        {
            float maxResult = 0;
            float result = 0;
            float dist = 0;

            for(int i = 0; i < needInSight.Count; ++i)
            {
                dist = (needInSight[i] - transform.position).magnitude;
                result = (Traits[1].Value - dist) / Traits[1].Value;

                if(result > maxResult)
                {
                    maxResult = result;
                }
            }
            return maxResult;
        }

        private double[] ComputePerception()
        {
            GetSenseRefresh();

            CurrentFear = EvaluateFear();
            inputs[0] = CurrentHunger / Traits[2].Value;
            inputs[1] = CurrentLife / Traits[3].Value;
            inputs[2] = CurrentFear / Traits[4].Value;
            inputs[3] = CurrentReproductionNeed / 100;
            inputs[4] = CurrentWaterNeed / Traits[2].Value;

            inputs[5] = EvaluateNeed(foodPosTemp);
            inputs[6] = EvaluateNeed(waterPosTemp);

            return inputs;
        }

        #endregion

        // EXECUTION ***************************************************************************************************
        #region Execution

        private void UpdatePassiveNeedsIncrease()
        {
            //PASSIVE NEEDS INCREASE

            CurrentReproductionNeed += Time.deltaTime * 0.01f;
            CurrentHunger += Time.deltaTime * 0.02f;
            CurrentWaterNeed += Time.deltaTime * 0.03f;
        }

        public void Update()
        {
            if (IsAlive)
            {
                UpdatePassiveNeedsIncrease();

                rateTimer += Time.deltaTime;
                if (rateTimer > Traits[0].Value)
                {
                    ExecuteDecision(geneticBrain.Compute(ComputePerception()));
                    rateTimer = 0;
                }

            }
        }
        #endregion

        // NeuralNetwork Computes Controller parameters to output a decision with is controlled here ******************
        #region Decisions
        public override void ExecuteDecision(double[] inputs)
        {
            int decisionIndex = JNNMath.MaxIndex(inputs);
            switch (decisionIndex)
            {
                case 0:
                    SearchForFood();
                    break;
                case 1:
                    Eat();
                    break;
                case 2:
                    SearchForWater();
                    break;
                case 3:
                    Drink();
                    break;
                case 4:
                    SearchForPartner();
                    break;
                case 5:
                    Reproduct();
                    break;
                case 6:
                    Attack();
                    break;
                case 7:
                    Flee();
                    break;
                case 8:
                    Wait();
                    break;
            }
        }

        #endregion

        // ACTIONS ****************************************************************************************************
        #region Actions

        public enum CurrentAction
        {
            SearchForFood,
            Eat,
            SearchForWater,
            Drink,
            SearchForPartner,
            Reproduct,
            Attack,
            Flee,
            Wait,
        }
        public CurrentAction CurrentlyDoing;

        public void SearchForFood() // Could be a prey or vegetables
        {
            CurrentlyDoing = CurrentAction.SearchForFood;
        }

        public void Eat()
        {
            CurrentlyDoing = CurrentAction.Eat;

        }

        public void SearchForWater()
        {
            CurrentlyDoing = CurrentAction.SearchForWater;

        }

        public void Drink()
        {
            CurrentlyDoing = CurrentAction.Drink;

        }

        public void SearchForPartner()
        {
            CurrentlyDoing = CurrentAction.SearchForPartner;

        }

        public override void Reproduct()
        {
            if (potentialPartner != null)
            {
                if (potentialPartner.AskReproduction(this) && AskReproduction(potentialPartner))
                {
                    Debug.LogError(this.UniqueID + " and " + potentialPartner.UniqueID + " are reproducing");
                    CurrentlyDoing = CurrentAction.Reproduct;
                    potentialPartner.CurrentlyDoing = CurrentAction.Reproduct;

                    if (Gender == GenderType.Male)
                    {
                        evolutionManager.Request_ComputeReproduction(this, potentialPartner);
                    }
                    else
                    {
                        evolutionManager.Request_ComputeReproduction(potentialPartner, this);
                    }
                }
            }
        }

        public void Attack()
        {
            CurrentlyDoing = CurrentAction.Attack;

        }

        public void Flee()
        {
            CurrentlyDoing = CurrentAction.Flee;

        }

        public void Wait()
        {
            CurrentlyDoing = CurrentAction.Wait;

        }
        #endregion

        // DIE ********************************************************************************************************
        #region Die
        public override void Die()
        {
            IsAlive = false;

            evolutionManager.GetEvaluationData(ComputeEvaluationData());
        }

        public override GeneticEvaluationData ComputeEvaluationData()
        {
            GeneticEvaluationData data = new GeneticEvaluationData();
            data.instanceID = UniqueID;

            SurvivedTime = Time.realtimeSinceStartup - TimeAtBorn;

            double[] evaluate = new double[] // we evaluate fitness on DNA Parameters AND meta-parameters 
            {
                SurvivedTime,
                FoodEaten,
                NumberOfChilds,
            };

            return data;
        }
        #endregion

        // REPRODUCTION : Here is a method to create child instances of two Entities **********************************
        #region ReproductionAndMutation
        public Controller_Animal potentialPartner;

        public override bool AskReproduction(GeneticInstanceController fromPartner)
        {
            if (CurrentlyDoing == CurrentAction.SearchForPartner)
            {
                Controller_Animal partner = fromPartner as Controller_Animal;
                if (Gender != partner.Gender)
                {
                    return true; //TODO later
                }
            }
            return false;
        }

        public override void MutateGene(Gene gene)
        {
            throw new NotImplementedException();
        }
               
        #endregion
    }
}
