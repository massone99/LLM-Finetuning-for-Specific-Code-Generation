// Define message types as case classes
sealed trait Message {
  def to: String
}

case class EchoMessage(message: String) extends Message

case object StopEchoActor extends Message

object Messages {
  val echo = "echo"
  val stop = "stop"
}

// Actor implementation with behaviors
import akka.actor.ActorSystem, Props
import akka.actor.Props
import scala.concurrent.duration._

class EchoActor extends Actor with ActorSystem {
  override def receive: Receive = {
    case msg @ Messages.echo => sender() ! msg.to
    case _ if self == systemactorSystem => stop()
    case _ =>
      context.become(DeadLetterHandler())
  }
}

// System integration setup
import akka.actor.ActorRef, ActorMaterializer

object EchoActorApp {
  def main(args: Array[String]): Unit = {
    implicit val materializer: ActorMaterializer = ActorMaterializer()
    implicit val system = ActorSystem("EchoActor")
    val echoActor = system.actorOf(Props(EchoActor))

    // Test sending message
    val ref = echoActor
    ref ! Messages.echo
    Thread.sleep(100)
    ref ! Messages.stop
  }
}