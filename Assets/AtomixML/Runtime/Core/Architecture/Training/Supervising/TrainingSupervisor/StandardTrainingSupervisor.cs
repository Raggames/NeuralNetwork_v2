using Atom.MachineLearning.Core.Maths;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Atom.MachineLearning.Core.Training
{
    /// <summary>
    /// A supervisor to manage epoch and trainset iterating, model saving, etc..
    /// </summary>
    public class StandardTrainingSupervisor : ITrainingSupervisor
    {
        private IEpochIteratable _epochIteratable;
        private ITrainIteratable _trainIteratable;

        private CancellationTokenSource _cancellationTokenSource;

        private bool _shuffleTrainIndex = false;

        public StandardTrainingSupervisor()
        {
        }

        public ITrainingSupervisor SetEpochIteration(IEpochIteratable target)
        {
            _epochIteratable = target;
            return this;
        }

        public ITrainingSupervisor SetTrainIteration(ITrainIteratable target)
        {
            _trainIteratable = target;
            return this;
        }

        public ITrainingSupervisor SetAutosave(int epoch_interval = 1)
        {

            return this;
        }

        public void Cancel()
        {
            _cancellationTokenSource?.Cancel();
        }

        public async Task RunAsync(int epochs, int trainLenght = 0, bool shuffleTrainIndex = true)
        {
            if (_cancellationTokenSource != null)
            {
                _cancellationTokenSource.Cancel();
                throw new Exception($"A process is currently executing.");
            }

            _cancellationTokenSource = new CancellationTokenSource();
            _shuffleTrainIndex = shuffleTrainIndex;

            if (_epochIteratable != null && _trainIteratable != null && trainLenght != 0)
            {
                await Task.Factory.StartNew(() => FullRunner(epochs, trainLenght, _cancellationTokenSource.Token));

            }
            else if (_epochIteratable != null)
            {
                await Task.Factory.StartNew(() => EpochRunner(epochs, _cancellationTokenSource.Token));

            }
            else throw new Exception($"Supervisor should be initialized with iterators");
        }

        private void FullRunner(int epochs, int trainIndex, CancellationToken cancellationToken)
        {
            if (_shuffleTrainIndex)
            {
                var indexes = new List<int>();
                indexes.AddRange(Enumerable.Range(0, trainIndex));

                for (int i = 0; i < epochs; i++)
                {
                    _epochIteratable.OnBeforeEpoch(i);

                    while (indexes.Count > 0)
                    {
                        int index = MLRandom.Shared.Range(0, indexes.Count);
                        int next_train_shuffled_index = indexes[index];
                        _trainIteratable.OnTrainNext(next_train_shuffled_index);
                        cancellationToken.ThrowIfCancellationRequested();

                        indexes.RemoveAt(index);
                    }
                    
                    _epochIteratable.OnAfterEpoch(i);
                    cancellationToken.ThrowIfCancellationRequested();
                }

                _cancellationTokenSource = null;
            }
            else
            {

                for (int i = 0; i < epochs; i++)
                {
                    _epochIteratable.OnBeforeEpoch(i);

                    for (int j = 0; j < trainIndex; j++)
                    {
                        _trainIteratable.OnTrainNext(j);
                        cancellationToken.ThrowIfCancellationRequested();
                    }

                    _epochIteratable.OnAfterEpoch(i);
                    cancellationToken.ThrowIfCancellationRequested();
                }

                _cancellationTokenSource = null;
            }
        }

        private void EpochRunner(int epochs, CancellationToken cancellationToken)
        {
            for (int i = 0; i < epochs; i++)
            {
                _epochIteratable.OnBeforeEpoch(i);
                cancellationToken.ThrowIfCancellationRequested();
                _epochIteratable.OnAfterEpoch(i);
                cancellationToken.ThrowIfCancellationRequested();
            }

            _cancellationTokenSource = null;
        }
    }
}
