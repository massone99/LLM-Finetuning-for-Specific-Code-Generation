import akka.actor.{Actor, ActorRef, ActorSystem, OneForOneStrategy, Props, SupervisorStrategy}
import akka.actor.SupervisorStrategy._

import scala.concurrent.duration._

// Define the Child Actor
class FaultyChildActor extends Actor {
  def receive: Receive = {
    case "causeNull" => throw new NullPointerException("Simulated NullPointerException")
    case msg        => println(s"Child Actor received: $msg")
  }
}

// Define the EscalatingSupervisor Actor
class EscalatingSupervisor extends Actor {
  override val supervisorStrategy: SupervisorStrategy = OneForOneStrategy() {
    case _: NullPointerException => println("Escalating Supervisor: Escalating NullPointerException")
    case _: Exception => println("Escalating Supervisor: Escalating unknown exception")
  }

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

  supervisor! "Hello"
  supervisor! "causeNull"
  supervisor! "World"
}