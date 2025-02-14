import akka.actor.{Actor, ActorRef, ActorSystem, OneForOneStrategy, Props, SupervisorStrategy}
import akka.actor.SupervisorStrategy.{Resume, Escalate}
import scala.concurrent.duration._

// Define the FaultyChildActor
class FaultyChildActor extends Actor {
  def receive: Receive = {
    case "causeNull" =>
      println(s"${self.path.name} received 'causeNull' and will throw NullPointerException.")
      throw new NullPointerException("Null pointer exception")
    case msg =>
      println(s"${self.path.name} received message: $msg")
  }
}

// Define the EscalatingSupervisor Actor
class EscalatingSupervisor extends Actor {
  override val supervisorStrategy: SupervisorStrategy = OneForOneStrategy() {
    case _: NullPointerException =>
      println("EscalatingSupervisor: Resuming child after NullPointerException.")
      Resume
    case _: Exception =>
      println("EscalatingSupervisor: Escalating failure to supervisor.")
      Escalate
  }

  // Create the child actor
  val child: ActorRef = context.actorOf(Props[FaultyChildActor](), "faultyChildActor")

  def receive: Receive = {
    case msg =>
      child.forward(msg)
  }
}

// Usage Example (for testing purposes)
object EscalatingSupervisorApp extends App {
  val system = ActorSystem("EscalatingSupervisorSystem")
  val supervisor = system.actorOf(Props[EscalatingSupervisor](), "escalatingSupervisor")

  // Send a message that causes NullPointerException
  supervisor! "causeNull"

  // Allow some time for processing before shutdown
  Thread.sleep(1000)
  system.terminate()
}