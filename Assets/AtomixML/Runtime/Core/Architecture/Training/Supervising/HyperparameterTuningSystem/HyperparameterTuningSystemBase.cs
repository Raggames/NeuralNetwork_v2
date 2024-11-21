using Atom.MachineLearning.Core.Training;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.Core
{
    public abstract class HyperparameterTuningSystemBase<KModelTuningProfile, KHyperParameterSet, TTrainer, TModel, TModelInput, TModelOutput>
            where KHyperParameterSet : IHyperParameterSet
            where KModelTuningProfile : ITuningProfile<KHyperParameterSet>
            where TModel : IMLModel<TModelInput, TModelOutput>
            where TTrainer : IMLTrainer<TModel, TModelInput, TModelOutput>
    {
        /// <summary>
        /// Start search using trainers in parallel, executing a search by accessing tuning profile of each trainer.
        /// Assume that trainer implement the given tuning profile
        /// </summary>
        /// <typeparam name="KModelTuningProfile"> The model tuning profile </typeparam>
        /// <typeparam name="KHyperParameterSet"> The parameter set for the given tuning profile </typeparam>
        /// <typeparam name="TModel"> the model type </typeparam>
        /// <typeparam name="TModelInput"> model input </typeparam>
        /// <typeparam name="TModelOutput"> model output </typeparam>
        /// <param name="trainers"> the batch of trainers that will be used in parallel to search the hyperparameter dimensions </param>
        /// <param name=""></param>
        /// <returns> the best found set of hyperparameter data </returns>
        public abstract Task<KHyperParameterSet> Search(int iterations, KModelTuningProfile kModelTuningProfile, TModelInput[] t_inputs, TTrainer[] trainers);

    }
}
