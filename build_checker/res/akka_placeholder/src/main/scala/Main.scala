import akka.actor.{Actor, ActorRef, ActorSystem, Props, OneForOneStrategy, SupervisorStrategy}
import akka.actor.SupervisorStrategy.{Restart, Escalate}
import scala.concurrent.duration._
import scala.concurrent.ExecutionContext.Implicits.global

// Define the ChildActor
class ChildActor extends Actor {
  def receive: Receive = {
    case "causeNull" => throw new NullPointerException("Null pointer exception triggered!")
    case msg => println(s"ChildActor received: $msg")
  }
}

// Define the FaultyChildActor
class FaultyChildActor extends Actor {
  def receive: Receive = {
    case "failNow" => throw new NullPointerException("FaultyChildActor failing now!")
    case msg => println(s"FaultyChildActor received: $msg")
  }
}

// Define the EscalatingSupervisor
class EscalatingSupervisor extends Actor {
  override val supervisorStrategy = OneForOneStrategy() {
    case _: NullPointerException =>
      println("EscalatingSupervisor: Escalating failure to parent supervisor.")
      Escalate
  }

  // Create the child actor
  val faultyChild: ActorRef = context.actorOf(Props[FaultyChildActor](), "faultyChildActor")

  def receive: Receive = {
    case msg =>
      println(s"EscalatingSupervisor sending message: $msg")
      faultyChild! msg
  }
}

// Usage Example (for testing purposes)
object EscalatingSupervisorApp extends App {
  val system = ActorSystem("EscalatingSupervisorSystem")
  val supervisor = system.actorOf(Props[EscalatingSupervisor](), "escalatingSupervisor")

  // Send messages to the child actor through the supervisor
  supervisor! "causeNull"
  supervisor! "failNow"
  supervisor! "Another message"
  supervisor! "causeNull" // Should escalate again
  supervisor! "failNow"    // Should escalate once more

  // Allow some time for processing before shutdown
  Thread.sleep(1000)
  system.terminate()
}