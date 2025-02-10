import akka.actor.{Actor, ActorRef, ActorSystem, OneForOneStrategy, Props, SupervisorStrategy, Terminated}
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
  override val supervisorStrategy: SupervisorStrategy = OneForOneStrategy(
    maxNrOfRetries = 3,
    withinTimeRange = 1.minute
  ) {
    case _: NullPointerException =>
      println("EscalatingSupervisor encountered a NullPointerException. Escaling failure.")
      SupervisorStrategy.Escalate
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

  // Send messages to the child actor
  supervisor! "causeNull"   // Should throw NullPointerException
  supervisor! "Hello"     // Should pass without error
  supervisor! "causeNull"   // Should throw NullPointerException again
  supervisor! "Goodbye"     // Should pass without error

  // Allow some time for processing before shutdown
  Thread.sleep(2000)
  system.terminate()
}