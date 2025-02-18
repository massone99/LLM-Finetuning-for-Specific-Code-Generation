import akka.actor.{Actor, ActorSystem, Props}
import akka.routing.Props

case object CauseNull

class FaultyChildActor extends Actor {
  def receive = {
    case "causeNull" => throw new NullPointerException("Message caused null")
    case _ => println("Received non-null message")
  }
}

class EscalatingSupervisor extends Actor {
  val child = context.actorOf(Props[FaultyChildActor], "faultyChild")

  def receive = {
    case msg: Any => child forward msg
  }

  override val supervisorStrategy =
    OneForOneStrategy()(context.system) {
      case e: NullPointerException => Escalate
    }
}

object Main extends App {
  implicit val system = ActorSystem("SupervisionSystem")
  val supervisor = system.actorOf(Props[EscalatingSupervisor], "supervisor")

  supervisor ! CauseNull

  Thread.sleep(1000)
}