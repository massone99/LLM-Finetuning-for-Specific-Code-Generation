import akka.actor.{Actor, ActorSystem, OneForOneStrategy, SupervisionStrategy, Props}
import scala.concurrent.duration._

// Define the EscalatingSupervisor Actor
class EscalatingSupervisor extends Actor {
  var supervisorStrategy = OneForOneStrategy() withDefaultStrategy // Initialize to one-for-one
  def receive: Receive = {
    case msg =>
      if (supervisorStrategy == OneForOneStrategy()) {
        println(s"EscalatingSupervisor received $msg; default strategy engaged.")
        supervisorStrategy = SupervisorStrategy.resume() // Switch to resume strategy
      } else if (supervisorStrategy == SupervisorStrategy.resume()) {
        println("EscalatingSupervisor: Resume strategy engaged.")
        val esc = context.parent // Get parent for supervision escalation
        esc.forward(this, "faultyChild")
      } else {
        println("Supervisor is not configured with a strategy.")
      }
  }
}

// Define the FaultyChildActor that fails when receiving "causeNull"
class FaultyChildActor extends Actor {
  def receive: Receive = {
    case msg =>
      if (msg == "causeNull") {
        throw new NullPointerException("FaultyChildActor: Cause Null Exception")
      } else {
        println(s"FaultyChildActor processed: $msg")
      }
    case _ => println("FaultyChildActor processed unknown message.")
  }
}

// Usage Example for EscalatingSupervisor and FaultyChildActor
object SupervisorFailureEscalationApp extends App {
  val system = ActorSystem("FailureEscalatorSystem")

  // EscalatingSupervisor as Supervisor Actor
  val escSupervisor = system.actorOf(Props[EscalatingSupervisor](), "escalator")

  // FaultyChildActor under EscalatingSupervisor supervision
  val childActor = escSupervisor? "faultyChild"

  // Allow some time for the actor to fail
  Thread.sleep(1500)

  // Check for the expected failure after a short delay
  system.terminate()
}