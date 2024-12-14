import akka.actor.{Actor, ActorRef, ActorSystem, OneForOneStrategy, Props, SupervisorStrategy}
import akka.actor.SupervisorStrategy._
import scala.concurrent.duration._

// Define the FaultyChildActor
class FaultyChildActor extends Actor {
  def receive: Receive = {
    case "causeNull" =>
      println(s"${self.path.name} received 'causeNull' and will throw NullPointerException.")
      throw new NullPointerException("Simulated NullPointerException.")
    case msg =>
      println(s"${self.path.name} received message: $msg")
  }
}

// Define the EscalatingSupervisor Actor
class EscalatingSupervisor extends Actor {
  override val supervisorStrategy: SupervisorStrategy = OneForOneStrategy() {
    case _: NullPointerException =>
      println("EscalatingSupervisor: Escalating the failure.")
      Escalate
  }

  // Create the child actor
  val child: ActorRef = context.actorOf(Props[FaultyChildActor](), "faultyChildActor")

  def receive: Receive = {
    case msg =>
      child.forward(msg)
  }
}

// Define the TopLevelSupervisor Actor to handle escalation
class TopLevelSupervisor extends Actor {
  override val supervisorStrategy: SupervisorStrategy = OneForOneStrategy() {
    case _: NullPointerException =>
      println("TopLevelSupervisor: Stopping the failing child.")
      Stop
    case _: Exception =>
      println("TopLevelSupervisor: Restarting the failing child.")
      Restart
  }

  // Create the EscalatingSupervisor as child
  val escalatingSupervisor: ActorRef = context.actorOf(Props[EscalatingSupervisor](), "escalatingSupervisor")

  def receive: Receive = {
    case msg =>
      escalatingSupervisor.forward(msg)
  }
}

// Usage Example (for testing purposes)
object EscalationApp extends App {
  val system = ActorSystem("EscalationSystem")
  val topSupervisor = system.actorOf(Props[TopLevelSupervisor](), "topSupervisor")

  // Send a message that causes the child to throw an exception
  topSupervisor ! "causeNull"

  // Allow some time for supervision to take effect before shutdown
  Thread.sleep(1000)
  system.terminate()
}