import akka.actor.{Actor, ActorRef, ActorSystem, Props, OneForOneStrategy, SupervisorStrategy}
import akka.actor.SupervisorStrategy.{Resume, Escalate}
import scala.concurrent.duration._

// Define the FaultyChildActor
class FaultyChildActor extends Actor {
  def receive: Receive = {
    case "causeNull" =>
      println("FaultyChildActor: Throwing NullPointerException...")
      throw new NullPointerException("Null pointer exception.")
    case msg => println(s"FaultyChildActor: Received -> $msg")
  }
}

// Define the EscalatingSupervisor
class EscalatingSupervisor extends Actor {
  override val supervisorStrategy: SupervisorStrategy = OneForOneStrategy(
    maxNrOfRetries = 3,
    withinTimeRange = 1.minute
  ) {
    case _: NullPointerException =>
      println("EscalatingSupervisor: Escalating supervisor strategy due to NullPointerException.")
      Resume
    case _ =>
      println("EscalatingSupervisor: Escalating failure with another strategy.")
      Escalate
  }

  // Create the child actor
  val child: ActorRef = context.actorOf(Props[FaultyChildActor](), "faultyChild")

  def receive: Receive = {
    case msg => child.forward(msg)
  }
}

// Usage Example (for testing purposes)
object EscalationHierarchyApp extends App {
  val system = ActorSystem("EscalationHierarchySystem")

  val supervisor = system.actorOf(Props[EscalatingSupervisor](), "escalatingSupervisor")

  // Send messages to the child actor
  supervisor! "Hello Child"
  supervisor! "causeNull"   // Should trigger NullPointerException and escalate
  supervisor! "After Escalation"

  // Allow some time for processing before shutdown
  Thread.sleep(2000)
  system.terminate()
}