use log::info;
use std::cmp::min_by;
use std::num::NonZeroUsize;
use std::sync::mpsc;
use std::thread::{self, JoinHandle};

/// A ThreadPool that manages a variable number of threads.
/// The maximum number of threads however cannot exceed the number of available threads on the system.
///
/// # Note
/// When you are done with the thread pool, you must call `ThreadPool::terminate`.
/// This will ensure that all threads are terminated.
pub struct ThreadPool {
    /// Vector of worker threads
    workers: Vec<Worker>,
    /// Channel to send jobs to the threads
    pub senders: Vec<mpsc::Sender<Message>>,
    // Channel to receive thread state from the threads
    pub state_receiver: mpsc::Receiver<IdleState>,
}

impl ThreadPool {
    /// Create a new ThreadPool.
    ///
    /// The size is the number of threads in the pool.
    ///
    /// # Panics
    ///
    /// The `new` function will panic if the size is zero.
    pub fn new(size: usize) -> ThreadPool {
        // panic if size 0
        assert!(size > 0, "Size must be greater than 0");

        // min between os available threads and size
        let size = number_of_threads_to_use(size);
        let mut senders = Vec::with_capacity(size);

        let (state_sender, state_receiver) = mpsc::channel::<IdleState>();
        let mut workers = Vec::with_capacity(size);

        for id in 0..size {
            let (job_sender, job_receiver) = mpsc::channel::<Message>();
            // create some threads and store them in the vector
            workers.push(Worker::new(id, job_receiver, state_sender.clone()));
            senders.push(job_sender.clone());
        }

        //let state_receiver = Arc::new(state_receiver);

        ThreadPool {
            workers,
            senders,
            state_receiver,
        }
    }

    /// Execute a function in the thread pool.
    /// The function will be executed in one of the threads in the pool.
    pub fn execute<F>(&self, f: F, id: usize)
    where
        F: FnOnce() + Send + 'static,
    {
        let job = Message::NewJob(Box::new(f));

        self.senders[id].send(job).unwrap();
    }

    /// Terminate the thread pool.
    /// By calling this method, the thread pool will be dropped.
    pub fn terminate(_: Self) {}
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        let workers_len = self.workers.len();
        for i in 0..workers_len {
            self.senders[i].send(Message::Terminate).unwrap();
        }
        for w in &mut self.workers {
            if let Some(thread) = w.thread.take() {
                thread.join().unwrap();
            }
        }
    }
}

struct Worker {
    _id: usize,
    thread: Option<JoinHandle<()>>,
}

impl Worker {
    fn new(
        id: usize,
        receiver: mpsc::Receiver<Message>,
        sender: mpsc::Sender<IdleState>,
    ) -> Worker {
        let thread = thread::spawn(move || {
            // send idle state to main thread
            sender.send(IdleState { id }).unwrap();
            // receive jobs from main thread
            loop {
                let message = receiver.recv().unwrap();

                match message {
                    Message::NewJob(job) => {
                        info!("Worker {} got a job; executing.", id);
                        job();
                        sender.send(IdleState { id }).unwrap();
                    }
                    Message::Terminate => {
                        info!("Worker {} was told to terminate.", id);
                        break;
                    }
                }
            }
        });

        Worker {
            _id: id,
            thread: Some(thread),
        }
    }
}

type Job = Box<dyn FnOnce() + Send + 'static>;

/// Returns the number of threads to use, based on the desired size and the number of available threads.
fn number_of_threads_to_use(desired_size: usize) -> usize {
    min_by(
        thread::available_parallelism().unwrap(),
        NonZeroUsize::new(desired_size).unwrap(),
        |x: &NonZeroUsize, y: &NonZeroUsize| x.cmp(y),
    )
    .get()
}

pub enum Message {
    Terminate,
    NewJob(Job),
}

/// Thread signal for when it is idle
pub struct IdleState {
    pub id: usize,
}

#[cfg(test)]
mod tests {
    use log::error;

    use super::*;
    use std::time::Duration;

    #[test]
    fn it_works() {
        let pool = ThreadPool::new(4);

        let (tx, rx) = mpsc::channel::<usize>();

        let mut i = 0;
        loop {
            if i == 4 {
                break;
            }
            let idle_thread = pool.state_receiver.recv();
            match &idle_thread {
                Ok(_) => {}
                Err(e) => {
                    error!("Error: {}", e);
                    break;
                }
            };
            i += 1;
            let idle_thread = idle_thread.unwrap();
            let idle_thread_id = idle_thread.id;

            let tx = tx.clone();

            pool.execute(
                move || {
                    println!("Thread {} is working", idle_thread_id);
                    thread::sleep(Duration::from_secs(1));
                    tx.send(idle_thread_id).unwrap();
                },
                idle_thread_id,
            )
        }

        for _ in 0..4 {
            let thread_id = rx.recv().unwrap();
            println!("Thread {} sent", thread_id);
            assert!(thread_id < 4);
        }

        ThreadPool::terminate(pool);
    }

    #[test]
    fn test_request_more_threads_than_available() {
        let available_threads = thread::available_parallelism().unwrap().get();
        let pool = ThreadPool::new(available_threads + 1);

        assert_eq!(pool.workers.len(), available_threads);
        pool.state_receiver.recv().unwrap();
        ThreadPool::terminate(pool);
    }

    #[test]
    fn test_request_less_threads_than_available() {
        let available_threads = thread::available_parallelism().unwrap().get();
        let pool = ThreadPool::new(available_threads - 1);

        assert_eq!(pool.workers.len(), available_threads - 1);
        ThreadPool::terminate(pool);
    }

    #[test]
    fn test_request_zero_threads() {
        // test panics
        std::panic::catch_unwind(|| {
            ThreadPool::new(0);
        })
        .expect_err("Should panic");
    }
}
