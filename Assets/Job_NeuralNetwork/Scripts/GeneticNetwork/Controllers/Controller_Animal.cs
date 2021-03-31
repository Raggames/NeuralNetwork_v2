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
        [Header("Entity DNA Traits")]
        public GenderType Gender;
        public enum GenderType
        {
            Female,
            Male,
        }

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

        public float TimeAtBorn;

        [Header("Brain Executions")]
        protected double[] inputs;
        protected double[] outputs;


        // ************************************************************************************************************
        public override void Init(GeneticBrain GeneticEntity, GeneticEvolutionManager EvolutionManager, List<Gene> DnaTraits)
        {
            base.Init(GeneticEntity, EvolutionManager, DnaTraits);

            RandGender();
            if(DnaTraits != null)
            {
                Traits = DnaTraits;
            }
            else
            {
                // Randomizing by a delta on all traits value to get some chaos in individutes at start
                for(int i = 0; i < Traits.Count; ++i)
                {
                    Traits[i] = new Gene(0, Traits[i].TraitName, RandomizeByDelta(Traits[i].Value, 0.1f), RandomizeByDelta(Traits[i].Dominance, 0.05f));
                }
                ThinkRate = new Gene(0, ThinkRate.TraitName, RandomizeByDelta(ThinkRate.Value, 0.1f), RandomizeByDelta(ThinkRate.Dominance, 0.05f));
            }
        }

        public override void Born()
        {
            IsAlive = true;
            TimeAtBorn = Time.realtimeSinceStartup;
        }

        private void RandGender()
        {
            float rand = UnityEngine.Random.Range(0f, 1f);
            if(rand > 0.5f)
            {
                Gender = GenderType.Female;
            }
            else
            {
                Gender =  GenderType.Male;
            }
        }

        private float RandomizeByDelta(float input, float delta)
        {
            float rand = UnityEngine.Random.Range(-delta, delta);
            return input + rand;
        }

        // PERCEPTION *************************************************************************************************
        #region Perception

        private double[] ComputePerception()
        {
            return new double[1];
        }

        #endregion

        // EXECUTION ***************************************************************************************************
        #region Execution
        public void Update()
        {
            if (IsAlive)
            {
                rateTimer += Time.deltaTime;
                if (rateTimer > ThinkRate.Value)
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
            if (potentialPartner.AskReproduction(this) && AskReproduction(potentialPartner))
            {
                Debug.LogError(this.UniqueID + " and " + potentialPartner.UniqueID + " are reproducing");
                CurrentlyDoing = CurrentAction.Reproduct;
                potentialPartner.CurrentlyDoing = CurrentAction.Reproduct;

                if(Gender == GenderType.Male)
                {
                    evolutionManager.Request_ComputeReproduction(this, potentialPartner);
                }
                else
                {
                    evolutionManager.Request_ComputeReproduction(potentialPartner, this);
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
            if(CurrentlyDoing == CurrentAction.SearchForPartner)
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
