import akka.actor.{Actor, ActorSystem, Props}
import akka.routing.RoundRobinPool

case class Work(job: String)

class WorkerActor extends Actor {
  def receive: Receive = {
    case Work(job) =>
      println(s"${self.path.name} working on: $job")
    case _ =>
      println(s"${self.path.name} received an unknown message.")
  }
}

object RouterActorApp extends App {
  val system = ActorSystem("RouterSystem")

  // Create a router with 3 workers
  val router = system.actorOf(
    RoundRobinPool(3).props(Props[WorkerActor]()),
    "workerRouter"
  )

  // Send multiple tasks
  for (i <- 1 to 5) {
    router ! Work(s"Task-$i")
  }

  Thread.sleep(500)
  system.terminate()
}