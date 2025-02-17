import akka.actor.{Actor, ActorLogging, Props, SupervisorStrategy, Terminated}
import akka.routing.RoundRobinPool

class EscalatingSupervisor extends Actor with ActorLogging {
  val faultyChildActor = context.actorOf(Props[FaultyChildActor], "faultyChild")

  override def supervisorStrategy: SupervisorStrategy =
    OneForOneStrategy() {
      case _: NullPointerException => Escalate
    }

  override def receive: Receive = {
    case msg: String =>
      log.info(s"EscalatingSupervisor received message: $msg")
      faultyChildActor forward s"causeNull_$msg"
    case Terminated(child) =>
      log.error("Faulty child actor terminated, restarting...")
      context.watch(context.actorOf(Props[FaultyChildActor], "faultyChild"))
  }
}

class FaultyChildActor extends Actor with ActorLogging {
  override def receive: Receive = {
    case "causeNull" => throw new NullPointerException("Simulating null pointer")
    case msg: String =>
      log.info(s"FaultyChildActor received message: $msg")
  }
}

object EscalatingSupervisorApp extends App {
  import akka.actor.ActorSystem
  implicit val system = ActorSystem("EscalationExample")

  val supervisor = system.actorOf(Props[EscalatingSupervisor], "supervisor")

  // Simulate failure and observe the escalation behavior
  supervisor ! "test"
}