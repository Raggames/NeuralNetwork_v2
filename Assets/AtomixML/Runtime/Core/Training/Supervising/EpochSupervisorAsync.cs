using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Atom.MachineLearning.Core
{
    /// <summary>
    /// A simple supervisor to manage async epoch running and cancelation
    /// </summary>
    public class EpochSupervisorAsync
    {
        public delegate void IterationCallbackHandler(int epochIndex);
        public delegate void EndEpochsCallbacHandler();

        private readonly IterationCallbackHandler _iterationCallbackHandler;

        private CancellationTokenSource _cancellationTokenSource;

        public EpochSupervisorAsync(IterationCallbackHandler iterationCallbackHandler)
        {
            _iterationCallbackHandler = iterationCallbackHandler;
        }

        public async Task Run(int epochs)
        {
            if(_cancellationTokenSource != null)
            {
                _cancellationTokenSource.Cancel();
                throw new Exception($"A process is currently executing.");
            }
            _cancellationTokenSource = new CancellationTokenSource();

            await Task.Factory.StartNew(() => AsyncRunner(epochs), _cancellationTokenSource.Token);
        }

        public void Cancel()
        {
            _cancellationTokenSource?.Cancel();
        }

        private void AsyncRunner(int epochs)
        {
            for(int i = 0; i < epochs; i++)
            {
                _iterationCallbackHandler(i);
            }

            _cancellationTokenSource = null;
        }
    }
}
