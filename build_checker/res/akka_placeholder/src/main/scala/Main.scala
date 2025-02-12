import akka.actor.{Actor, ActorRef, ActorSystem, OneForOneStrategy, Props, SupervisorStrategy}
import akka.actor.SupervisorStrategy.{Restart, Escalate}
import scala.concurrent.duration._

// Define the FaultyChildActor
class FaultyChildActor extends Actor {
  def receive: Receive = {
    case "causeNull" =>
      println(s"${self.path.name} received 'causeNull', attempting to trigger NullPointerException.")
      throw new NullPointerException("Simulated NullPointerException")
    case msg =>
      println(s"${self.path.name} received message: $msg")
  }
}

// Define the EscalatingSupervisor Actor
class EscalatingSupervisor extends Actor {
  override val supervisorStrategy: SupervisorStrategy = OneForOneStrategy(
    maxNrOfRetries = 2,
    withinTimeRange = 1.minute
  ) {
    case _: NullPointerException =>
      println(s"EscalatingSupervisor caught NullPointerException, escalating failure.")
      Escalate
    case other =>
      println(s"EscalatingSupervisor encountered an unexpected error: $other")
      Restart
  }

  // Create the child actor
  val child: ActorRef = context.actorOf(Props[FaultyChildActor](), "faultyChildActor")

  def receive: Receive = {
    case msg =>
      println(s"EscalatingSupervisor received message: $msg from child actor")
      child.forward(msg)
  }
}

// Usage Example (for testing purposes)
object EscalatingSupervisorApp extends App {
  val system = ActorSystem("EscalatingSupervisorSystem")
  val supervisor = system.actorOf(Props[EscalatingSupervisor](), "escalatingSupervisor")

  supervisor! "causeNull"
  supervisor! "Post-nullPointerException message"

  Thread.sleep(2000)
  system.terminate()
}