```scala
import akka.actor.{Actor, ActorLogging, ActorRef, ActorSystem, Props}
import akka.actor.SupervisorStrategy.{Escalate, Restart, Stop}
import akka.actor.OneForOneStrategy

class EscalatingSupervisor extends Actor with ActorLogging {
  val faultyChildActor: ActorRef = context.actorOf(Props[FaultyChildActor], "faultyChildActor")

  override def receive: Receive = {
    case "causeNull" => faultyChildActor ! "causeNull"
    case _ => log.info("Received unknown message")
  }

  override def supervisorStrategy: SupervisorStrategy = OneForOneStrategy(maxNrOfRetries = 10, withinTimeRange = 1 minute) {
    case _: NullPointerException => Escalate
  }
}

class FaultyChildActor extends Actor with ActorLogging {
  override def receive: Receive = {
    case "causeNull" => throw new NullPointerException("FaultyChildActor encountered a NullPointerException")
    case _ => log.info("Received unknown message")
  }
}

object EscalatingSupervisorDemo extends App {
  val system = ActorSystem("EscalatingSupervisorDemo")
  val escalatingSupervisor = system.actorOf(Props[EscalatingSupervisor], "escalatingSupervisor")

  escalatingSupervisor ! "causeNull"
}
```

This code defines an Akka Supervisor Actor named EscalatingSupervisor that supervises a child actor named FaultyChildActor. The FaultyChildActor throws a NullPointerException when it receives a "causeNull" message. The supervisor's strategy is set to escalate the failure to its own supervisor upon encountering an exception. The supervision hierarchy and failure escalation are demonstrated in the EscalatingSupervisorDemo object.