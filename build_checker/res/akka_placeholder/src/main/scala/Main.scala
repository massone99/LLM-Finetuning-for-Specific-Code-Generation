import akka.actor.{Actor, ActorRef, ActorSystem, Props, OneForOneStrategy}
import akka.actor.SupervisorStrategy.{Resume, Escalate}
import scala.concurrent.duration._

class FaultyChildActor extends Actor {
  def receive: Receive = {
    case "causeNull" => throw new NullPointerException("Null pointer exception caused by message: causeNull")
    case msg => println(s"FaultyChildActor received message: $msg")
  }
}

class EscalatingSupervisor extends Actor {
  override val supervisorStrategy: OneForOneStrategy = OneForOneStrategy() {
    case _: NullPointerException => Resume
    case _: Exception => Escalate
  }
  
  val child: ActorRef = context.actorOf(Props[FaultyChildActor](), "faultyChild")
  def receive: Receive = {
    case msg => child.forward(msg)
  }
}

object EscalatingSupervisorApp extends App {
  val system = ActorSystem("EscalatingSupervisorSystem")
  val supervisor = system.actorOf(Props[EscalatingSupervisor](), "escalatingSupervisor")
  supervisor! "causeNull"
  supervisor! "failAfterEscalate" // message to demonstrate escalation and failure
  Thread.sleep(1000)
  system.terminate()
}

// Note: The failure escalation might take some time (up to 1 second in this case) because the child needs to throw the exception before the supervisor can escalate it.
//