import akka.actor.{Actor, ActorRef, ActorSystem, OneForOneStrategy, Props, SupervisorStrategy, Terminated}
import akka.actor.SupervisorStrategy._

import scala.concurrent.duration._

// Define the ChildActor
class FaultyChildActor extends Actor {
  def receive: Receive = {
    case "causeNull" =>
      throw new NullPointerException("ChildActor received null cause.")
    case msg =>
      println(s"ChildActor received message: $msg")
  }
}

// Define the EscalatingSupervisor Actor
class EscalatingSupervisor extends Actor {
  override val supervisorStrategy: SupervisorStrategy = OneForOneStrategy(
    maxNrOfRetries = 3,
    withinTimeRange = 1.minute
  ) {
    case _: NullPointerException =>
      println("EscalatingSupervisor: Encountered NullPointerException. Escalating failure.")
      SupervisorStrategy.Escape
    case _: Exception =>
      SupervisorStrategy.None
  }

  // Create the child actor
  val child: ActorRef = context.actorOf(Props[FaultyChildActor](), "childActor")

  def receive: Receive = {
    case msg =>
      child.forward(msg)
  }
}

// Usage Example (for testing purposes)
object EscalatingSupervisorApp extends App {
  val system = ActorSystem("EscalatingSupervisorSystem")
  val supervisor = system.actorOf(Props[EscalatingSupervisor](), "supervisor")

  supervisor! "causeNull"
  supervisor! "Post-null message"
}